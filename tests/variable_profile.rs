extern crate menoh;

mod utils;
use utils::constant;

#[test]
fn add_input_profile_success() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims = utils::get_input_dims();

    assert!(
        vpt_builder
            .add_input_profile(
                constant::INPUT_VARIABLE_NAME,
                menoh::Dtype::Float,
                &input_dims
            )
            .is_ok()
    );
    assert!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .is_ok()
    );
}

#[test]
fn add_input_profile_fail_with_invalid_name() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims = utils::get_input_dims();

    vpt_builder
        .add_input_profile("Invalid", menoh::Dtype::Float, &input_dims)
        .unwrap();
    assert_eq!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .err()
            .unwrap(),
        menoh::Error::VariableNotFound
    );
}

#[test]
fn add_input_profile_fail_with_invalid_dtype() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();

    let input_dims: Vec<i32> = vec![0, 0, 0, 0];

    vpt_builder
        .add_input_profile(
            constant::INPUT_VARIABLE_NAME,
            menoh::Dtype::Float,
            &input_dims,
        )
        .unwrap();

    assert_eq!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .err()
            .unwrap(),
        menoh::Error::DimensionMismatch
    );
}

#[test]
fn add_output_profile_success() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims = utils::get_input_dims();

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
        .build_variable_profile_table(&model_data)
        .unwrap();
}
