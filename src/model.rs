use std;
use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::mem;

use libc;
use libc::c_void;

use error::{cvt_r, Error};
use ffi;
use model_data::ModelData;
use variable_profile::{Dtype, VariableProfileTable};

#[derive(Clone, Copy)]
pub struct Buffer<'a> {
    ptr: *mut c_void,
    size: usize,
    len: usize,
    dtype: Dtype,
    _phantom: PhantomData<&'a c_void>,
}

pub struct ModelBuilder<'a, 's> {
    handle: ffi::menoh_model_builder_handle,
    external_bufs: HashMap<&'s str, Buffer<'a>>,
}

pub struct Model<'a, 's> {
    external_bufs: HashMap<&'s str, Buffer<'a>>,
    handle: ffi::menoh_model_handle,
}

impl<'a> Buffer<'a> {
    pub fn new<T>(buf: &'a mut [T], dtype: Dtype) -> Self {
        let buf_size = mem::size_of::<T>() * buf.len();
        Buffer {
            ptr: buf.as_mut_ptr() as *mut c_void,
            size: buf_size,
            len: buf.len(),
            dtype,
            _phantom: PhantomData,
        }
    }

    pub fn update<T>(&mut self, buf: &mut [T]) -> Result<(), Error> {
        let len_as_byte = mem::size_of::<T>() * buf.len();
        if len_as_byte != self.size {
            // TODO: error type
            return Err(Error::StdError);
        }
        unsafe {
            libc::memcpy(self.ptr, buf.as_ptr() as *const c_void, self.size);
        }
        Ok(())
    }

    pub fn as_slice<T>(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
}

impl<'a, 's> ModelBuilder<'a, 's> {
    pub fn new(variable_profile_table: &VariableProfileTable) -> Result<Self, Error> {
        let mut handle: ffi::menoh_model_builder_handle = unsafe { mem::uninitialized() };
        cvt_r(|| unsafe {
            ffi::menoh_make_model_builder(
                variable_profile_table.get_handle(),
                &mut handle as *mut ffi::menoh_model_builder_handle,
            )
        })?;
        Ok(ModelBuilder {
            handle,
            external_bufs: HashMap::new(),
        })
    }

    // TODO: Varidate external buffer size
    pub fn attach_external_buffer(
        &mut self,
        name: &'s str,
        buffer: Buffer<'a>,
    ) -> Result<(), Error> {
        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;
        cvt_r(|| unsafe {
            ffi::menoh_model_builder_attach_external_buffer(
                self.handle,
                name_c_expr.as_ptr(),
                buffer.ptr,
            )
        })?;
        self.external_bufs.insert(name, buffer);
        Ok(())
    }

    pub fn build_model(
        &self,
        model_data: &ModelData,
        backend_name: &str,
        backend_config: &str,
    ) -> Result<Model, Error> {
        let backend_name = CString::new(backend_name).map_err(|_| Error::VariableNotFound)?;
        let backend_config = CString::new(backend_config).map_err(|_| Error::VariableNotFound)?;
        let mut model_handle: ffi::menoh_model_handle = unsafe { mem::uninitialized() };
        cvt_r(|| unsafe {
            ffi::menoh_build_model(
                self.handle,
                model_data.get_handle(),
                backend_name.as_ptr(),
                backend_config.as_ptr(),
                &mut model_handle as *mut ffi::menoh_model_handle,
            )
        })?;
        Ok(Model {
            handle: model_handle,
            external_bufs: self.external_bufs.clone(),
        })
    }
}

impl<'a, 's> Model<'a, 's> {
    pub fn run(&mut self) -> Result<(), Error> {
        cvt_r(|| unsafe { ffi::menoh_model_run(self.handle) })?;
        Ok(())
    }

    pub fn get_attached_buffer(&self, name: &str) -> Result<Buffer<'a>, Error> {
        self.external_bufs
            .get(name)
            .ok_or(Error::VariableNotFound)
            .map(|buf| *buf)
    }

    pub fn get_internal_buffer(&self, name: &str) -> Result<Buffer, Error> {
        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;
        let dtype = self.get_dtype(&name_c_expr)?;
        let len = self.get_variable_len(&name_c_expr)?;

        let ptr = self.get_buf_handle(&name_c_expr)?;
        // TODO: Think on dtype handling
        let size = match dtype {
            Dtype::Float => len * mem::size_of::<f32>(),
        };

        Ok(Buffer {
            ptr,
            size,
            len,
            dtype,
            _phantom: PhantomData,
        })
    }

    fn get_dtype(&self, name: &CString) -> Result<Dtype, Error> {
        let mut dtype = ffi::menoh_dtype::default();
        cvt_r(|| unsafe {
            ffi::menoh_model_get_variable_dtype(
                self.handle,
                name.as_ptr(),
                &mut dtype as *mut ffi::menoh_dtype,
            )
        })?;
        Ok(Dtype::from(dtype))
    }

    fn get_variable_len(&self, name: &CString) -> Result<usize, Error> {
        let dims_size = self.get_variable_dims_size(name)?;

        let mut dims = Vec::new();
        for i in 0..dims_size {
            dims.push(self.get_variable_dims_at(name, i)?)
        }
        Ok(dims.iter().fold(1, |acc, x| acc * x) as usize)
    }

    fn get_variable_dims_size(&self, name: &CString) -> Result<i32, Error> {
        let mut dims_size = libc::int32_t::default();
        cvt_r(|| unsafe {
            ffi::menoh_model_get_variable_dims_size(
                self.handle,
                name.as_ptr(),
                &mut dims_size as *mut libc::int32_t,
            )
        })?;
        Ok(dims_size)
    }

    fn get_variable_dims_at(&self, name: &CString, index: i32) -> Result<i32, Error> {
        let mut dst_dim = libc::int32_t::default();
        cvt_r(|| unsafe {
            ffi::menoh_model_get_variable_dims_at(
                self.handle,
                name.as_ptr(),
                index as i32,
                &mut dst_dim as *mut libc::int32_t,
            )
        })?;
        Ok(dst_dim)
    }

    fn get_buf_handle(&self, name: &CString) -> Result<*mut c_void, Error> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        cvt_r(|| unsafe {
            ffi::menoh_model_get_variable_buffer_handle(
                self.handle,
                name.as_ptr(),
                &mut handle as *mut *mut c_void,
            )
        })?;
        Ok(handle)
    }
}

impl<'a, 's> Drop for ModelBuilder<'a, 's> {
    fn drop(&mut self) {
        unsafe { ffi::menoh_delete_model_builder(self.handle) }
    }
}

impl<'a, 's> Drop for Model<'a, 's> {
    fn drop(&mut self) {
        unsafe { ffi::menoh_delete_model(self.handle) }
    }
}
