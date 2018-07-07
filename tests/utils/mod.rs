pub mod constant;

pub fn get_input_dims() -> Vec<i32> {
    vec![
        constant::INPUT_BATCH_SIZE,
        constant::INPUT_CHANNEL_NUM,
        constant::INPUT_WIDTH,
        constant::INPUT_HEIGHT,
    ]
}
