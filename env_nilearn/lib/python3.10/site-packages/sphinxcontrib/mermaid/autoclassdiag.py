import inspect

from sphinx.errors import ExtensionError
from sphinx.util import import_object

from .exceptions import MermaidError


def get_classes(*cls_or_modules, strict=False):
    """
    given one or several fully qualified names, yield class instances found.

    If ``strict`` is only consider classes that are strictly defined in that module
    and not imported from somewhere else.
    """
    for cls_or_module in cls_or_modules:
        try:
            obj = import_object(cls_or_module)
        except ExtensionError as e:
            raise MermaidError(str(e))

        if inspect.isclass(obj):
            yield obj

        elif inspect.ismodule(obj):
            for obj_ in obj.__dict__.values():
                if inspect.isclass(obj_) and (not strict or obj_.__module__.startswith(obj.__name__)):
                    yield obj_
        else:
            raise MermaidError(f"{cls_or_module} is not a class nor a module")


def class_diagram(*cls_or_modules, full=False, strict=False, namespace=None):
    inheritances = set()

    def get_tree(cls):
        for base in cls.__bases__:
            if base.__name__ == "object":
                continue
            if namespace and not base.__module__.startswith(namespace):
                continue
            inheritances.add((base.__name__, cls.__name__))
            if full:
                get_tree(base)

    for cls in get_classes(*cls_or_modules, strict=strict):
        get_tree(cls)

    if not inheritances:
        return ""

    return "classDiagram\n" + "\n".join(f"  {a} <|-- {b}" for a, b in sorted(inheritances))


if __name__ == "__main__":

    class A:
        pass

    class B(A):
        pass

    class C1(B):
        pass

    class C2(B):
        pass

    class D(C1, C2):
        pass

    class E(C1):
        pass

    print(class_diagram("__main__.D", "__main__.E", full=True))
