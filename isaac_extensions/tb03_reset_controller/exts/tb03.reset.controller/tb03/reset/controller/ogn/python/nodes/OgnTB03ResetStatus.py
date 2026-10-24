import omni.graph.core as og

from tb03.reset.controller.controller import get_controller


class OgnTB03ResetStatus:
    @staticmethod
    def compute(db):
        controller = get_controller()
        complete = controller is not None and controller.consume_completion()
        db.outputs.success = complete
        db.outputs.execOut = (
            og.ExecutionAttributeState.ENABLED if complete else og.ExecutionAttributeState.DISABLED
        )
        return True
