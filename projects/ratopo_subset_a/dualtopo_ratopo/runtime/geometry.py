"""Native first-layer anchor bootstrap, followed by predicted-centerline PE."""
from ratopo_centerline_decoder import CenterlineGeometryRouter

GEOMETRY_MARKER = 'selected_native_first_layer_bootstrap_next_five_PE'


class NativeBootstrapRouter(CenterlineGeometryRouter):
    def __init__(self, choice, pc_range):
        if choice != GEOMETRY_MARKER:
            raise ValueError('This project requires its native anchor bootstrap')
        self.current = None
        self.pc_range = tuple(pc_range)
        self.inputs = []
        self.outputs = []
        if len(pc_range) != 6 or any(pc_range[i + 3] <= pc_range[i] for i in range(3)):
            raise ValueError('Positive native coordinate extents are required')

    def before_layer(self, layer, output, kwargs):
        if layer == 0:
            if self.inputs or self.current is not None or 'dualtopo_centerline_world' in kwargs:
                raise ValueError('The first layer must use native anchor initialization')
            self.inputs.append(None)
            return kwargs
        return super().before_layer(layer, output, kwargs)
