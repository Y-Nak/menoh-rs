//! This module provides "raw" declarations and linkage for `Menoh` C API.
//!
//! ***In nomarl use case, no need to use this module directly.***
//!
//! To get information of `Menoh` C API, please refer to [here](https://pfnet-research.github.io/menoh/)

use libc::{c_char, c_void, int32_t};

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct menoh_model_data {
    private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct menoh_model {
    private: [u8; 0],
}
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct menoh_model_builder {
    private: [u8; 0],
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct menoh_variable_profile_table {
    private: [u8; 0],
}
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct menoh_variable_profile_table_builder {
    private: [u8; 0],
}

#[allow(non_camel_case_types)]
pub type menoh_error_code = int32_t;
#[allow(non_camel_case_types)]
pub type menoh_dtype = int32_t;

#[allow(non_camel_case_types)]
pub type menoh_model_data_handle = *mut menoh_model_data;

#[allow(non_camel_case_types)]
pub type menoh_model_handle = *mut menoh_model;
#[allow(non_camel_case_types)]
pub type menoh_model_builder_handle = *mut menoh_model_builder;

#[allow(non_camel_case_types)]
pub type menoh_variable_profile_table_handle = *mut menoh_variable_profile_table;
#[allow(non_camel_case_types)]
pub type menoh_variable_profile_table_builder_handle = *mut menoh_variable_profile_table_builder;

#[allow(non_upper_case_globals)]
pub const menoh_error_code_success: menoh_error_code = 0;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_std_error: menoh_error_code = 1;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_unknown_error: menoh_error_code = 2;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_invalid_filename: menoh_error_code = 3;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_unsupported_onnx_opset_version: menoh_error_code = 4;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_onnx_parse_error: menoh_error_code = 5;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_invalid_dtype: menoh_error_code = 6;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_invalid_attribute_type: menoh_error_code = 7;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_unsupported_operator_attribute: menoh_error_code = 8;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_dimension_mismatch: menoh_error_code = 9;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_variable_not_found: menoh_error_code = 10;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_index_out_of_range: menoh_error_code = 11;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_json_parse_error: menoh_error_code = 12;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_invalid_backend_name: menoh_error_code = 13;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_unsupported_operator: menoh_error_code = 14;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_failed_to_configure_operator: menoh_error_code = 15;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_backend_error: menoh_error_code = 16;
#[allow(non_upper_case_globals)]
pub const menoh_error_code_same_named_variable_already_exist: menoh_error_code = 17;

#[allow(non_upper_case_globals)]
pub const menoh_dtype_float: menoh_dtype = 0;

#[link(name = "menoh")]
extern "C" {
    pub fn menoh_get_last_error_message() -> *const c_char;

    pub fn menoh_make_model_data_from_onnx(
        onnx_filename: *const c_char,
        dst_handle: *mut menoh_model_data_handle,
    ) -> menoh_error_code;
    pub fn menoh_delete_model_data(model_data: menoh_model_data_handle);
    pub fn menoh_model_data_optimize(
        model_data: menoh_model_data_handle,
        variable_profile_table: menoh_variable_profile_table_handle,
    ) -> menoh_error_code;

    pub fn menoh_make_variable_profile_table_builder(
        builder: *mut menoh_variable_profile_table_builder_handle,
    ) -> menoh_error_code;
    pub fn menoh_delete_variable_profile_table_builder(
        builder: menoh_variable_profile_table_builder_handle,
    );
    pub fn menoh_variable_profile_table_builder_add_input_profile_dims_2(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const c_char,
        dtype: menoh_dtype,
        num: int32_t,
        size: int32_t,
    ) -> menoh_error_code;
    pub fn menoh_variable_profile_table_builder_add_input_profile_dims_4(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const c_char,
        dtype: menoh_dtype,
        num: int32_t,
        channel: int32_t,
        height: int32_t,
        width: int32_t,
    ) -> menoh_error_code;
    pub fn menoh_variable_profile_table_builder_add_output_profile(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const c_char,
        dtype: menoh_dtype,
    ) -> menoh_error_code;
    pub fn menoh_build_variable_profile_table(
        builder: menoh_variable_profile_table_builder_handle,
        model_data: menoh_model_data_handle,
        dst_handle: *mut menoh_variable_profile_table_handle,
    ) -> menoh_error_code;
    pub fn menoh_delete_variable_profile_table(
        variable_profile_table: menoh_variable_profile_table_handle,
    );
    pub fn menoh_variable_profile_table_get_dtype(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const c_char,
        dst_dtype: *mut menoh_dtype,
    ) -> menoh_error_code;
    pub fn menoh_variable_profile_table_get_dims_size(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const c_char,
        dst_size: *mut int32_t,
    ) -> menoh_error_code;
    pub fn menoh_variable_profile_table_get_dims_at(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const c_char,
        index: int32_t,
        dst_size: *mut int32_t,
    ) -> menoh_error_code;

    pub fn menoh_make_model_builder(
        variable_profile_table: menoh_variable_profile_table_handle,
        dst_handle: *mut menoh_model_builder_handle,
    ) -> menoh_error_code;
    pub fn menoh_delete_model_builder(handle: menoh_model_builder_handle);
    pub fn menoh_model_builder_attach_external_buffer(
        builder: menoh_model_builder_handle,
        variable_name: *const c_char,
        buffer_handle: *mut c_void,
    ) -> menoh_error_code;
    pub fn menoh_build_model(
        builder: menoh_model_builder_handle,
        model_data: menoh_model_data_handle,
        backend_name: *const c_char,
        backend_config: *const c_char,
        dst_model_handle: *mut menoh_model_handle,
    ) -> menoh_error_code;
    pub fn menoh_delete_model(model: menoh_model_handle);
    pub fn menoh_model_get_variable_buffer_handle(
        model: menoh_model_handle,
        variable_name: *const c_char,
        dst_data: *mut *mut c_void,
    ) -> menoh_error_code;
    pub fn menoh_model_get_variable_dtype(
        model: menoh_model_handle,
        variable_name: *const c_char,
        dst_dtype: *mut menoh_dtype,
    ) -> menoh_error_code;
    pub fn menoh_model_get_variable_dims_size(
        model: menoh_model_handle,
        variable_name: *const c_char,
        dst_size: *mut int32_t,
    ) -> menoh_error_code;
    pub fn menoh_model_get_variable_dims_at(
        model: menoh_model_handle,
        variable_name: *const c_char,
        index: int32_t,
        dst_size: *mut int32_t,
    ) -> menoh_error_code;
    pub fn menoh_model_run(model: menoh_model_handle) -> menoh_error_code;
}
