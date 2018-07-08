//! This module contains ModelData
//!
//!
use std;
use std::ffi::CString;
use std::path::Path;

use error::{cvt_r, Error};
use ffi;
use ffi::menoh_model_data_handle;
use variable_profile::VariableProfileTable;

/// Represent model data defined by ONNX file.
pub struct ModelData {
    handle: menoh_model_data_handle,
}

impl ModelData {
    /// Create ModelData from given ONNX file.
    pub fn new<P>(onnx_path: P) -> Result<Self, Error>
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

    /// Optimize model data
    pub fn optimize(&mut self, vpt: &VariableProfileTable) -> Result<(), Error> {
        cvt_r(|| unsafe { ffi::menoh_model_data_optimize(self.handle, vpt.get_handle()) })?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");

    fn get_model_path(name: &str) -> PathBuf {
        let mut path = PathBuf::new();
        path.push(ROOT_DIR);
        path.push("tests");
        path.push("resource");
        path.push(name);
        path
    }

    #[test]
    fn load_onnx_success() {
        let model_path = get_model_path("mnist.onnx");
        assert!(ModelData::new(model_path).is_ok());
    }

    #[test]
    fn load_onnx_fail_with_wrong_path() {
        assert_matches!(ModelData::new("").err().unwrap(), Error::InvalidFileName);
    }

    #[test]
    fn load_onnx_fail_with_broken_string() {
        assert_matches!(
            ModelData::new("p\0ath").err().unwrap(),
            Error::InvalidFileName
        );
    }

    #[test]
    fn load_onnx_fail_with_wrong_format() {
        let invalid_model_path = get_model_path("invalid_onnx.onnx");
        assert_matches!(
            ModelData::new(invalid_model_path).err().unwrap(),
            Error::ONNXParseError
        )
    }
}
