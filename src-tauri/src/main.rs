// Windows release 版不弹控制台
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    z_biz_tool_math_lib::run()
}
