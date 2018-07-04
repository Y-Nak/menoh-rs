//! This module contains monoh model related.

use std;
use std::collections::HashMap;
use std::ffi::CString;
use std::marker::PhantomData;
use std::{mem, slice};

use libc;
use libc::c_void;

use dtype::{Dtype, DtypeCompatible};
use error::{cvt_r, Error};
use ffi;
use Buffer;
use ModelData;
use VariableProfileTable;

/// Builder of model.
///
/// Mainly used for buffer control.
pub struct ModelBuilder<'a, 's> {
    handle: ffi::menoh_model_builder_handle,
    external_bufs: HashMap<&'s str, RawBuffer<'a>>,
}

/// Main struct running inference.
///
/// An instance of `Model` is built by `ModelBuilder`
///
/// Lifetime of `Model` instance is bounded by attached buffer internal data if user attached external buffer.
pub struct Model<'a, 's> {
    external_bufs: HashMap<&'s str, RawBuffer<'a>>,
    handle: ffi::menoh_model_handle,
}

#[derive(Clone)]
struct RawBuffer<'a> {
    pub data: *mut c_void,
    len: usize,
    _phantom: PhantomData<&'a c_void>,
}

impl<'a, 's> ModelBuilder<'a, 's> {
    pub fn new(vpt: &VariableProfileTable) -> Result<Self, Error> {
        let mut handle: ffi::menoh_model_builder_handle = unsafe { mem::uninitialized() };
        cvt_r(|| unsafe {
            ffi::menoh_make_model_builder(
                vpt.get_handle(),
                &mut handle as *mut ffi::menoh_model_builder_handle,
            )
        })?;
        Ok(ModelBuilder {
            handle,
            external_bufs: HashMap::new(),
        })
    }

    /// Attach external buffer to the [`Model`][Model] generated from this instance.
    ///
    /// If user doesn't attach external buffer, then internal buffer is automatically generaged
    /// inside [`Model`][Model].
    ///
    /// [`Model`][Model] lifetime is bounded by buffer internal data.
    ///
    /// If user can ensure buffer dtype and size is valid, and [`VariableProfileTable`][VPT] is out of
    /// scope, there is an unsafe version of this method,
    /// [`attach_external_buffer_unchecked`][attach_external_buffer_unchecked], which doesn't
    /// require [`VariableProfileTable`][VPT] and skip validation against passed buffer.
    ///
    /// [VPT]: struct.VariableProfileTable.html
    /// [Model]: struct.Model.html
    /// [attach_external_buffer_unchecked]:
    /// struct.ModelBuilder.html#method.attach_external_buffer_unchecked
    pub fn attach_external_buffer<T>(
        &mut self,
        name: &'s str,
        buffer: &Buffer<'a, T>,
        vpt: &VariableProfileTable,
    ) -> Result<(), Error>
    where
        T: DtypeCompatible,
    {
        let variable_profile = vpt.get_variable_profile(name)?;

        if buffer.as_slice().len() < variable_profile.dims.iter().fold(1, |acc, x| acc * x) as usize
        {
            return Err(Error::InvalidBufferSize);
        }

        if !variable_profile.dtype.is_compatible::<T>() {
            return Err(Error::InvalidDtype);
        }

        unsafe { self.attach_external_buffer_unchecked(name, buffer) }
    }

    /// Attach external buffer to the [`Model`][Model] generated from this instance.
    ///
    /// If user doesn't attach external buffer, then internal buffer is automatically generaged
    /// inside model.
    ///
    /// [`Model`][Model] lifetime is bounded by buffer internal data.
    ///
    /// See the safe version, [`attach_external_buffer`][attach_external_buffer], for more
    /// infomation.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check that passed buffer is valid dtype and
    /// have enough size.
    ///
    /// User must ensure [`Dtype`][Dtype] and size of internal buffer size on oneself.
    ///
    /// [attach_external_buffer]: struct.ModelBuilder.html#method.attach_external_buffer
    /// [Model]: struct.Model.html
    /// [Dtype]: enum.Dtype.html
    ///
    pub unsafe fn attach_external_buffer_unchecked<T>(
        &mut self,
        name: &'s str,
        buffer: &Buffer<'a, T>,
    ) -> Result<(), Error>
    where
        T: DtypeCompatible,
    {
        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;
        let raw_buffer = RawBuffer::from(buffer);
        cvt_r(|| {
            ffi::menoh_model_builder_attach_external_buffer(
                self.handle,
                name_c_expr.as_ptr(),
                raw_buffer.data,
            )
        })?;
        self.external_bufs.insert(name, raw_buffer);
        Ok(())
    }

    /// Build model.
    ///
    /// After calling `build_model`, User can drop the instance of `ModelBuilder` in arbitary timing.
    pub fn build_model(
        &self,
        model_data: &ModelData,
        backend_name: &str,
        backend_config: &str,
    ) -> Result<Model<'a, 's>, Error> {
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
    /// Run model inference
    pub fn run(&mut self) -> Result<(), Error> {
        cvt_r(|| unsafe { ffi::menoh_model_run(self.handle) })?;
        Ok(())
    }

    /// Get reference to attached buffer.
    ///
    /// The reference to the buffer can live longer than this instance.
    pub fn get_attached_buffer<T>(&self, name: &str) -> Result<&'a [T], Error>
    where
        T: DtypeCompatible,
    {
        let raw_buf = self.external_bufs.get(name).ok_or(Error::VariableNotFound)?;
        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;

        self.validate_dtype::<T>(&name_c_expr)?;

        Ok(raw_buf.as_slice())
    }

    /// Get reference to attached buffer.
    ///
    /// The reference to the buffer can live longer than this instance.
    pub fn get_attached_buffer_mut<T>(&self, name: &str) -> Result<&'a mut [T], Error>
    where
        T: DtypeCompatible,
    {
        let raw_buf = self.external_bufs.get(name).ok_or(Error::VariableNotFound)?;
        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;

        self.validate_dtype::<T>(&name_c_expr)?;

        Ok(raw_buf.as_mut_slice())
    }

    /// Get reference to buffer generated inside model.
    ///
    /// The reference lifetime is bounded by this instance.
    pub fn get_internal_buffer<T>(&self, name: &str) -> Result<&[T], Error>
    where
        T: DtypeCompatible,
    {
        if self.external_bufs.contains_key(name) {
            return Err(Error::NotInternalBuffer);
        }

        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;

        self.validate_dtype::<T>(&name_c_expr)?;

        let len = self.get_variable_len(&name_c_expr)?;
        let ptr = self.get_buf_handle(&name_c_expr)?;
        unsafe { Ok(slice::from_raw_parts(ptr as _, len)) }
    }

    /// Get mutable reference to buffer generated inside model.
    ///
    /// The reference lifetime is bounded by this instance.
    pub fn get_internal_buffer_mut<T>(&self, name: &str) -> Result<&mut [T], Error>
    where
        T: DtypeCompatible,
    {
        if self.external_bufs.contains_key(name) {
            return Err(Error::NotInternalBuffer);
        }

        let name_c_expr = CString::new(name).map_err(|_| Error::VariableNotFound)?;

        self.validate_dtype::<T>(&name_c_expr)?;

        let len = self.get_variable_len(&name_c_expr)?;
        let ptr = self.get_buf_handle(&name_c_expr)?;
        unsafe { Ok(slice::from_raw_parts_mut(ptr as _, len)) }
    }

    /// Get dtype by name
    pub fn get_variable_dtype(&self, name: &str) -> Result<Dtype, Error> {
        let name = CString::new(name).map_err(|_| Error::VariableNotFound)?;
        self.get_variable_dtype_impl(&name)
    }

    /// Get dims by name
    pub fn get_variable_dims(&self, name: &str) -> Result<Vec<i32>, Error> {
        let name = CString::new(name).map_err(|_| Error::VariableNotFound)?;
        let dims_size = self.get_variable_dims_size(&name)?;

        let mut dims = Vec::new();
        for i in 0..dims_size {
            dims.push(self.get_variable_dims_at(&name, i)?)
        }
        Ok(dims)
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

    fn validate_dtype<T>(&self, name: &CString) -> Result<(), Error>
    where
        T: DtypeCompatible,
    {
        let dtype = self.get_variable_dtype_impl(&name)?;
        if dtype.is_compatible::<T>() {
            Ok(())
        } else {
            Err(Error::InvalidDtype)
        }
    }

    fn get_variable_dtype_impl(&self, name: &CString) -> Result<Dtype, Error> {
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
}

impl<'a> RawBuffer<'a> {
    fn from<T: DtypeCompatible>(buffer: &Buffer<'a, T>) -> Self {
        RawBuffer {
            data: buffer.as_slice().as_ptr() as _,
            len: buffer.as_slice().len(),
            _phantom: PhantomData,
        }
    }

    fn as_slice<T>(&self) -> &'a [T] {
        unsafe { slice::from_raw_parts(self.data as _, self.len) }
    }

    fn as_mut_slice<T>(&self) -> &'a mut [T] {
        unsafe { slice::from_raw_parts_mut(self.data as _, self.len) }
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
