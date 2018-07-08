extern crate menoh;

mod utils;
use utils::constant;

#[test]
fn optimize_success() {
    let mut model_data = menoh::ModelData::new(constant::MODEL_PATH).unwrap();
    let vpt_builder = utils::create_vpt_mock();
    assert!(model_data.optimize(&vpt_builder).is_ok());
}
