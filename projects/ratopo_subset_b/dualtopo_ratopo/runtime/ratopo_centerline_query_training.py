"""DualTopo RATopo inference components."""
import ast


import inspect


from ratopo_query_state import RATopoQueryFork


from ratopo_centerline_decoder import CenterlineFork, CenterlineBoundary, CenterlineGeometryRouter


class CenterlineQueryFork(CenterlineFork, RATopoQueryFork):
    """Keep the audited anchor/position associations and detached top66 cache."""
    pass


def export_query_boundary(boundary):
    return CenterlineBoundary(boundary.query, boundary.query_pos, boundary.anchors,
                              boundary.centerline_world, boundary.has_history)

