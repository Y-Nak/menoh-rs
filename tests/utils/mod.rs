extern crate menoh;

pub mod constant;

#[allow(dead_code)]
pub fn get_input_dims() -> Vec<i32> {
    vec![
        constant::INPUT_BATCH_SIZE,
        constant::INPUT_CHANNEL_NUM,
        constant::INPUT_WIDTH,
        constant::INPUT_HEIGHT,
    ]
}

#[allow(dead_code)]
pub fn create_vpt_mock(model_data: &menoh::ModelData) -> menoh::VariableProfileTable {
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

    vpt_builder
        .build_variable_profile_table(model_data)
        .unwrap()
}

#[allow(dead_code)]
pub fn get_input_buffer_size() -> usize {
    get_input_dims().iter().fold(1, |acc, val| acc * val) as usize
}

#[allow(dead_code)]
pub fn create_model_mock<'a>(
    model_data: &menoh::ModelData,
    allocated_mem: &'a mut Vec<f32>,
) -> menoh::Model<'a, 'static> {
    let buffer = menoh::Buffer::new(allocated_mem);

    let vpt = create_vpt_mock(&model_data);
    let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
    model_builder
        .attach_external_buffer(constant::INPUT_VARIABLE_NAME, &buffer, &vpt)
        .unwrap();
    model_builder
        .build_model(&model_data, menoh::Backend::MKL_DNN, "")
        .unwrap()
}
