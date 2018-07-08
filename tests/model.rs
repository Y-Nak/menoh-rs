extern crate menoh;
#[macro_use]
extern crate matches;

mod utils;
use utils::constant;

#[test]
fn model_buider_new() {
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt = utils::create_vpt_mock(&model_data);
    assert!(menoh::ModelBuilder::new(&vpt).is_ok());
}

#[test]
fn attach_external_buffer_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let buffer = menoh::Buffer::new(&mut allocated_mem);

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt = utils::create_vpt_mock(&model_data);
    let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
    assert!(
        model_builder
            .attach_external_buffer(constant::INPUT_VARIABLE_NAME, &buffer, &vpt)
            .is_ok()
    )
}

#[test]
fn attach_external_buffer_fail_with_already_exists() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let buffer = menoh::Buffer::new(&mut allocated_mem);

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt = utils::create_vpt_mock(&model_data);
    let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
    model_builder
        .attach_external_buffer(constant::INPUT_VARIABLE_NAME, &buffer, &vpt)
        .unwrap();

    assert_matches!(
        model_builder
            .attach_external_buffer(constant::INPUT_VARIABLE_NAME, &buffer, &vpt)
            .err()
            .unwrap(),
        menoh::Error::SameNamedVariableAlreadyExist
    );
}

#[test]
fn attach_external_buffer_fail_with_wrong_size() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); 1];
    let buffer = menoh::Buffer::new(&mut allocated_mem);

    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt = utils::create_vpt_mock(&model_data);
    let mut model_builder = menoh::ModelBuilder::new(&vpt).unwrap();
    assert_matches!(
        model_builder
            .attach_external_buffer(constant::INPUT_VARIABLE_NAME, &buffer, &vpt)
            .err()
            .unwrap(),
        menoh::Error::InvalidBufferSize
    )
}

#[test]
fn build_model_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();

    utils::create_model_mock(&model_data, &mut allocated_mem);
}

#[test]
fn run_model_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let mut model = utils::create_model_mock(&model_data, &mut allocated_mem);
    assert!(model.run().is_ok());
}

#[test]
fn get_input_buffer_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let model = utils::create_model_mock(&model_data, &mut allocated_mem);
    assert!(
        model
            .get_internal_buffer::<f32>(constant::OUTPUT_VARIABLE_NAME)
            .is_ok()
    );
}

#[test]
fn get_output_buffer_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let model = utils::create_model_mock(&model_data, &mut allocated_mem);
    assert!(
        model
            .get_attached_buffer::<f32>(constant::INPUT_VARIABLE_NAME)
            .is_ok()
    );
}

#[test]
fn get_buffer_fail_with_not_internal_buffer() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let model = utils::create_model_mock(&model_data, &mut allocated_mem);
    assert_matches!(
        model
            .get_internal_buffer::<f32>(constant::INPUT_VARIABLE_NAME)
            .err()
            .unwrap(),
        menoh::Error::NotInternalBuffer
    );
}

#[test]
fn get_variable_dims_from_model_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let model = utils::create_model_mock(&model_data, &mut allocated_mem);

    let dims = model
        .get_variable_dims(constant::INPUT_VARIABLE_NAME)
        .unwrap();
    assert_eq!(dims, utils::get_input_dims());
}

#[test]
#[allow(warnings)]
fn get_variable_dtype_from_model_success() {
    let mut allocated_mem: Vec<f32> = vec![Default::default(); utils::get_input_buffer_size()];
    let model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let model = utils::create_model_mock(&model_data, &mut allocated_mem);

    let dtype = model
        .get_variable_dtype(constant::INPUT_VARIABLE_NAME)
        .unwrap();
    assert_matches!(dtype, menoh::Dtype::Float);
}
