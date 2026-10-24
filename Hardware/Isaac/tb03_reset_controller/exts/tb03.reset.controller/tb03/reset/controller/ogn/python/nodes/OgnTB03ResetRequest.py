import omni.graph.core as og

from tb03.reset.controller.controller import get_controller


class OgnTB03ResetRequest:
    @staticmethod
    def compute(db):
        controller = get_controller()
        if controller is None:
            db.log_error("TB03 reset controller extension is not running")
            return False
        linear = db.inputs.linearVelocity
        angular = db.inputs.angularVelocity
        controller.queue(linear[0], linear[1], angular[2])
        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
