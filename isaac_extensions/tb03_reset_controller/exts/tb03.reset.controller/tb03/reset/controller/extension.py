"""Extension lifecycle only; OmniGraph handles custom-node registration."""

import omni.ext
import omni.kit.app

from .controller import ResetController, install_controller, remove_controller


class Extension(omni.ext.IExt):
    def on_startup(self, _ext_id: str) -> None:
        self._controller = ResetController()
        install_controller(self._controller)
        self._update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update
        )
        print("[TB03ResetController] Started; waiting for OmniGraph reset requests")

    def on_shutdown(self) -> None:
        self._update_subscription = None
        remove_controller()
        self._controller = None
        print("[TB03ResetController] Stopped")

    def _on_update(self, _event) -> None:
        self._controller.advance()
