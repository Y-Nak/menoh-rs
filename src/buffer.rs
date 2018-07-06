//! This module contains buffer, wrapping  menoh buffer control scheme.
//!

use dtype::DtypeCompatible;
use error::Error;

/// Buffer, a safe wrapper of menoh buffer control scheme.
///
/// It's highly recommended to control buffer content via this instance.
///
/// Lifetime of `Buffer` instance is bouded by internal data.
pub struct Buffer<'a, T>
where
    T: 'a + DtypeCompatible,
{
    data: &'a mut [T],
}

impl<'a, T> Buffer<'a, T>
where
    T: 'a + DtypeCompatible,
{
    /// Create buffer.
    ///
    /// * `data` - `data` can't be manipulate while `Buffer` instance lives.
    pub fn new(data: &'a mut [T]) -> Self {
        Buffer { data }
    }

    /// Update buffer content from other data.
    ///
    /// Data length must be same as internal data length.
    pub fn update(&mut self, data: &[T]) -> Result<(), Error> {
        if self.data.len() != data.len() {
            return Err(Error::InvalidBufferSize);
        }

        self.data.copy_from_slice(data);
        Ok(())
    }

    pub fn as_slice(&self) -> &[T] {
        self.data
    }
}
