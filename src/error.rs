//! This module defines Error and Result type of Menoh

use std::error;
use std::ffi::CStr;
use std::fmt;

use ffi;
use ffi::menoh_get_last_error_message;

#[derive(Debug)]
pub enum Error {
    // C-API Defined error
    StdError,
    UnknownError,
    InvalidFileName,
    UnsupportedONNXOpsetVersion,
    ONNXParseError,
    InvalidDtype,
    InvalidAttributeType,
    UnsupportedOperatorAttribute,
    DimensionMismatch,
    VariableNotFound,
    IndexOutOfRange,
    JsonParseError,
    InvalidBackendName,
    UnsupportedOperator,
    FailedToConfigureOperator,
    BackendError,
    SameNamedVariableAlreadyExist,

    // menoh-rs custom error
    InvalidBufferSize,
    NotInternalBuffer,
}

fn get_last_error_message<'a>() -> &'a str {
    unsafe {
        let s = CStr::from_ptr(menoh_get_last_error_message());
        match s.to_str() {
            Ok(s) => s,
            Err(_) => "Failed Conversion from C-String",
        }
    }
}

/// Convert error code defined in Menoh C API to Error.
///
/// * `ec` - If `ec` is NOT defined in `menoh::ffi`, then this function cause panic.
///
/// ***In normal use case, no need to use this function directly.***
pub fn cvt(ec: ffi::menoh_error_code) -> Result<ffi::menoh_error_code, Error> {
    match ec {
        ffi::menoh_error_code_success => Ok(ffi::menoh_error_code_success),
        ffi::menoh_error_code_std_error => Err(Error::StdError),
        ffi::menoh_error_code_unknown_error => Err(Error::UnknownError),
        ffi::menoh_error_code_invalid_filename => Err(Error::InvalidFileName),
        ffi::menoh_error_code_unsupported_onnx_opset_version => {
            Err(Error::UnsupportedONNXOpsetVersion)
        }
        ffi::menoh_error_code_onnx_parse_error => Err(Error::ONNXParseError),
        ffi::menoh_error_code_invalid_dtype => Err(Error::InvalidDtype),
        ffi::menoh_error_code_invalid_attribute_type => Err(Error::InvalidAttributeType),
        ffi::menoh_error_code_unsupported_operator_attribute => {
            Err(Error::UnsupportedOperatorAttribute)
        }
        ffi::menoh_error_code_dimension_mismatch => Err(Error::DimensionMismatch),
        ffi::menoh_error_code_variable_not_found => Err(Error::VariableNotFound),
        ffi::menoh_error_code_index_out_of_range => Err(Error::IndexOutOfRange),
        ffi::menoh_error_code_json_parse_error => Err(Error::JsonParseError),
        ffi::menoh_error_code_invalid_backend_name => Err(Error::InvalidBackendName),
        ffi::menoh_error_code_unsupported_operator => Err(Error::UnsupportedOperatorAttribute),
        ffi::menoh_error_code_failed_to_configure_operator => Err(Error::FailedToConfigureOperator),
        ffi::menoh_error_code_backend_error => Err(Error::BackendError),
        ffi::menoh_error_code_same_named_variable_already_exist => {
            Err(Error::SameNamedVariableAlreadyExist)
        }
        _ => unreachable!(),
    }
}

/// Convert error code defined in Menoh C API to Error.
///
/// * `f` - If error_code returned by `f` is NOT defined in `menoh::ffi`, then this function cause panic.
///
/// ***In normal use case, no need to use this function directly.***
pub fn cvt_r<F>(mut f: F) -> Result<ffi::menoh_error_code, Error>
where
    F: FnMut() -> ffi::menoh_error_code,
{
    cvt(f())
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let msg = match *self {
            Error::InvalidBufferSize => "Buffer size is invalid",
            Error::NotInternalBuffer => "Specified buffer is attached buffer",
            _ => get_last_error_message(),
        };
        write!(f, "{}", msg)
    }
}

impl error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cvt_to_ok() {
        let res = cvt(ffi::menoh_error_code_success);
        assert!(res.is_ok());
    }

    #[test]
    fn cvt_to_error() {
        let res = cvt(ffi::menoh_error_code_std_error);
        assert_matches!(res.err().unwrap(), Error::StdError);
    }

    #[test]
    #[should_panic]
    fn cvt_must_panic() {
        let _res = cvt(100);
    }
}
