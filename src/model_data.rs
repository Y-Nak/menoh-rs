//! This module contains ModelData
//!
//!
use std::ffi::CString;
use std::mem;
use std::path::Path;

use error::{cvt_r, Error, Result};
use ffi;
use ffi::menoh_model_data_handle;

pub struct ModelData {
    handle: menoh_model_data_handle,
}

impl ModelData {
    /// Create ModelData from given ONNX file.
    pub fn new(onnx_path: &Path) -> Result<Self> {
        let mut handle: menoh_model_data_handle = unsafe { mem::uninitialized() };
        let p = CString::new(onnx_path.to_str().ok_or(Error::InvalidFileName)?)
            .map_err(|_| Error::InvalidFileName)?;
        unsafe {
            cvt_r(|| {
                ffi::menoh_make_model_data_from_onnx(
                    p.as_ptr(),
                    &mut handle as *mut menoh_model_data_handle,
                )
            })?;
        }
        Ok(ModelData { handle })
    }

    #[doc(hidden)]
    pub fn get_handle(&self) -> ffi::menoh_model_data_handle {
        self.handle
    }

    /// Release internal allocated memory.
    ///
    /// User can call this method in arbitary timing after building model.
    ///
    /// If `ModelData` object goes out of scope, internal allocated memory is released
    /// automatically.
    ///
    /// ***If user call this function manually, Double release is NERVER occured***
    pub fn release(self) {}
}

impl Drop for ModelData {
    fn drop(&mut self) {
        unsafe { ffi::menoh_delete_model_data(self.handle) }
    }
}
