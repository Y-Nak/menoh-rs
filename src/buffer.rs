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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_buffer_success() {
        let mut v: Vec<f32> = vec![0., 1., 2.];
        let mut buffer = Buffer::new(&mut v);

        let v = vec![10., 11., 12.];
        buffer.update(&v).unwrap();
        assert_eq!(buffer.as_slice(), v.as_slice());
    }

    #[test]
    fn update_buffer_fail() {
        let mut v: Vec<f32> = vec![Default::default(); 3];
        let mut buffer = Buffer::new(&mut v);

        let v = vec![Default::default(); 4];
        assert_matches!(buffer.update(&v).err().unwrap(), Error::InvalidBufferSize);
    }
}
