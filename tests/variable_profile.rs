extern crate menoh;
#[macro_use]
extern crate matches;

mod utils;
use utils::constant;

#[test]
fn build_vpt_success() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    utils::create_vpt_mock(&model_data);
}

#[test]
fn add_input_profile_fail_with_invalid_name() {
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims = utils::get_input_dims();

    vpt_builder
        .add_input_profile("Invalid", menoh::Dtype::Float, &input_dims)
        .unwrap();

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    assert_matches!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .err()
            .unwrap(),
        menoh::Error::VariableNotFound
    );
}

#[test]
fn add_input_profile_fail_with_invalid_dims() {
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims: Vec<i32> = vec![0, 0, 0, 0];

    vpt_builder
        .add_input_profile(
            constant::INPUT_VARIABLE_NAME,
            menoh::Dtype::Float,
            &input_dims,
        )
        .unwrap();

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    assert_matches!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .err()
            .unwrap(),
        menoh::Error::DimensionMismatch
    );
}

#[test]
fn add_input_profile_fail_with_invalid_dims_len() {
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();
    let input_dims: Vec<i32> = vec![0, 0, 0, 0, 0];

    assert_matches!(
        vpt_builder
            .add_input_profile(
                constant::INPUT_VARIABLE_NAME,
                menoh::Dtype::Float,
                &input_dims
            )
            .err()
            .unwrap(),
        menoh::Error::DimensionMismatch
    );
}

#[test]
fn add_output_profile_fail_with_invalid_name() {
    let mut vpt_builder = menoh::VariableProfileTableBuilder::new().unwrap();

    vpt_builder
        .add_output_profile(constant::OUTPUT_VARIABLE_NAME, menoh::Dtype::Float)
        .unwrap();

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    assert_matches!(
        vpt_builder
            .build_variable_profile_table(&model_data)
            .err()
            .unwrap(),
        menoh::Error::VariableNotFound
    );
}

#[test]
#[allow(warnings)]
fn get_variable_profile_success() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt = utils::create_vpt_mock(&model_data);
    let input_profile = vpt.get_variable_profile(constant::INPUT_VARIABLE_NAME)
        .unwrap();
    assert_matches!(input_profile.dtype, menoh::Dtype::Float);
    assert_eq!(input_profile.dims, utils::get_input_dims());
}
