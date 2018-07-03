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

#[derive(Clone)]
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

impl<'a, 's> Drop for ModelBuilder<'a, 's> {
    fn drop(&mut self) {
        unsafe { ffi::menoh_delete_model_builder(self.handle) }
    }
}

//impl<'a, 's> Model<'a, 's> {
//    pub fn run(&mut self) -> Result<(), Error> {
//        cvt_r(|| unsafe { ffi::menoh_model_run(self.handle) })?;
//        Ok(())
//    }
//
//    pub fn get_attached_buffer(&self, name: &str) -> Option<&'a Buffer> {
//        self.external_bufs.get(name)
//    }
//
//    pub fn get_attached_buffer_mut(&mut self, name: &str) -> Option<&'a mut Buffer> {
//        self.external_bufs.get_mut(name)
//    }
//
//    // pub fn get_internal_buffer(&self, name: &str) -> Option<Buffer> {}
//    // pub fn get_internal_buffer_mut(&mut self, name: &str) -> Option<&mut Buffer> {}
//}
