// Launch at login through SMAppService.mainApp (the same login item the Swift
// shell registered, keyed by the bundle id com.panwudi.flyto-mlx).

use objc2_service_management::{SMAppService, SMAppServiceStatus};

fn service() -> objc2::rc::Retained<SMAppService> {
    // SAFETY: class method with no arguments.
    unsafe { SMAppService::mainAppService() }
}

pub fn enabled() -> bool {
    // SAFETY: plain property read.
    let status = unsafe { service().status() };
    status == SMAppServiceStatus::Enabled || status == SMAppServiceStatus::RequiresApproval
}

pub fn set(on: bool) -> Result<(), String> {
    let svc = service();
    // SAFETY: documented synchronous calls returning NSError on failure.
    let res = unsafe {
        if on { svc.registerAndReturnError() } else { svc.unregisterAndReturnError() }
    };
    res.map_err(|e| e.localizedDescription().to_string())
}
