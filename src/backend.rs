//! This module contains enum representing menoh backend.
//!

#[derive(Clone, Copy)]
pub enum Backend {
    #[allow(non_camel_case_types)]
    MKL_DNN,
}

impl Backend {
    #[doc(hidden)]
    pub fn value(&self) -> &str {
        match *self {
            Backend::MKL_DNN => "mkldnn",
        }
    }
}
