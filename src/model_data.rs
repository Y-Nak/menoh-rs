//! This module contains ModelData
//!
//!
use std;
use std::ffi::CString;
use std::path::Path;

use error::{cvt_r, Error};
use ffi;
use ffi::menoh_model_data_handle;

pub struct ModelData {
    handle: menoh_model_data_handle,
}

impl ModelData {
    /// Create ModelData from given ONNX file.
    pub fn new<P>(onnx_path: &P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut handle: menoh_model_data_handle = std::ptr::null_mut();
        let p = CString::new(onnx_path.as_ref().to_str().ok_or(Error::InvalidFileName)?)
            .map_err(|_| Error::InvalidFileName)?;
        cvt_r(|| unsafe {
            ffi::menoh_make_model_data_from_onnx(
                p.as_ptr(),
                &mut handle as *mut menoh_model_data_handle,
            )
        })?;
        Ok(ModelData { handle })
    }

    #[doc(hidden)]
    pub unsafe fn get_handle(&self) -> ffi::menoh_model_data_handle {
        self.handle
    }
}

impl Drop for ModelData {
    fn drop(&mut self) {
        unsafe { ffi::menoh_delete_model_data(self.handle) }
    }
}
