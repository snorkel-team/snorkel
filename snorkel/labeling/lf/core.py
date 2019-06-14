from typing import Any, Callable, Mapping, Optional, Tuple

from snorkel.types import DataPoint


class LabelingFunction:
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        """Base object for labeling functions, containing metadata and extra
        functionality
        Args:
          * name: name of the LF
          * f: function that implements the LF
          * label_space: set of labels the LF can output, including 0
          * schema: fields of the input DataPoints the LF needs
          * resources: labeling resources passed in to f via kwargs
          * fault_tolerant: output 0 if LF execution fails?
        """
        self.name = name
        self.label_space = label_space
        self.schema = schema
        self.fault_tolerant = fault_tolerant
        self._f = f
        self._resources = resources or {}

    def set_fault_tolerant(self, fault_tolerant: bool = True) -> None:
        self.fault_tolerant = fault_tolerant

    def __call__(self, x: DataPoint) -> int:
        if self.fault_tolerant:
            try:
                return self._f(x, **self._resources)
            except Exception:
                return 0
        return self._f(x, **self._resources)

    def __repr__(self) -> str:
        schema_str = f", DataPoint schema: {self.schema}" if self.schema else ""
        label_str = f", Label space: {self.label_space}" if self.label_space else ""
        return f"Labeling function {self.name}{schema_str}{label_str}"


class labeling_function:
    def __init__(
        self,
        name: Optional[str] = None,
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        fault_tolerant: bool = False,
    ) -> None:
        """Decorator to define a LabelingFunction object from a function

        Example usage:

        ```
        @labeling_function()
        def f(x: DataPoint) -> int:
            return 1 if x.a > 42 else 0
        print(f)  # "Labeling function f"

        @labeling_function(name="my_lf")
        def g(x: DataPoint) -> int:
            return 1 if x.a > 42 else 0
        print(g)  # "Labeling function my_lf"
        ```
        """
        self.name = name
        self.label_space = label_space
        self.schema = schema
        self.resources = resources
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        name = self.name or f.__name__
        return LabelingFunction(
            name=name,
            f=f,
            label_space=self.label_space,
            schema=self.schema,
            resources=self.resources,
            fault_tolerant=self.fault_tolerant,
        )
