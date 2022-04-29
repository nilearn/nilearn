import io
import base64
import urllib.parse


def figure_to_svg_bytes(fig):
    with io.BytesIO() as io_buffer:
        fig.savefig(
            io_buffer, format="svg", facecolor="white", edgecolor="white"
        )
        return io_buffer.getvalue()


def figure_to_svg_base64(fig):
    return base64.b64encode(figure_to_svg_bytes(fig)).decode()


def figure_to_svg_quoted(fig):
    return urllib.parse.quote(figure_to_svg_bytes(fig).decode("utf-8"))
