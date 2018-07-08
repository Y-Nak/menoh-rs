extern crate menoh;

pub mod constant;

pub fn get_input_dims() -> Vec<i32> {
    vec![
        constant::INPUT_BATCH_SIZE,
        constant::INPUT_CHANNEL_NUM,
        constant::INPUT_WIDTH,
        constant::INPUT_HEIGHT,
    ]
}

pub fn create_vpt_mock() -> menoh::VariableProfileTable {
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();

    let input_dims = get_input_dims();
    vpt_builder
        .add_input_profile(
            constant::INPUT_VARIABLE_NAME,
            menoh::Dtype::Float,
            &input_dims,
        )
        .unwrap();

    vpt_builder
        .add_output_profile(constant::OUTPUT_VARIABLE_NAME, menoh::Dtype::Float)
        .unwrap();

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    vpt_builder
        .build_variable_profile_table(&model_data)
        .unwrap()
}
