//! This module contains buffer attached to model
//!

use dtype::DtypeCompatible;
use error::Error;

/// Buffer that attached to model.
///
/// Lifetime of `Buffer` instance is bouded by internal data.
///
/// Buffer can attached to multiple models.
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
    /// Create buffer from a internal data.
    ///
    /// * `data` - `data` can't be manipulate while `Buffer` instance lives.
    pub fn new(data: &'a mut [T]) -> Self {
        Buffer { data }
    }

    /// Update Buffer from data
    ///
    /// Data length must be same as internal data length.
    /// **Caution: Internal data is also updated**
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

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}
