from typing import Any, Callable, List, Mapping, Optional, Tuple

from snorkel.types import DataPoint

from .preprocess import Preprocessor, PreprocessorMode


class LabelingFunction:
    def __init__(
        self,
        name: str,
        f: Callable[..., int],
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[Preprocessor]] = None,
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
          * preprocessors: list of Preprocessors to run on data points
          * fault_tolerant: output 0 if LF execution fails?
        """
        self.name = name
        self.label_space = label_space
        self.schema = schema
        self.fault_tolerant = fault_tolerant
        self._f = f
        self._resources = resources or {}
        self._preprocessors = preprocessors or []

    def set_fault_tolerant(self, fault_tolerant: bool = True) -> None:
        self.fault_tolerant = fault_tolerant

    def set_preprocessor_mode(self, mode: PreprocessorMode) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.set_mode(mode)

    def _preprocess_data_point(self, x: DataPoint) -> DataPoint:
        for preprocessor in self._preprocessors:
            x = preprocessor(x)
        return x

    def __call__(self, x: DataPoint) -> int:
        x = self._preprocess_data_point(x)
        if self.fault_tolerant:
            try:
                return self._f(x, **self._resources)
            except Exception:
                return 0
        return self._f(x, **self._resources)

    def __repr__(self) -> str:
        schema_str = f", DataPoint schema: {self.schema}" if self.schema else ""
        label_str = f", Label space: {self.label_space}" if self.label_space else ""
        preprocessor_str = f", Preprocessors: {self._preprocessors}"
        return f"Labeling function {self.name}{schema_str}{label_str}{preprocessor_str}"


class labeling_function:
    def __init__(
        self,
        name: Optional[str] = None,
        label_space: Optional[Tuple[int, ...]] = None,
        schema: Optional[Mapping[str, type]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        preprocessors: Optional[List[Preprocessor]] = None,
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
        self.preprocessors = preprocessors
        self.fault_tolerant = fault_tolerant

    def __call__(self, f: Callable[..., int]) -> LabelingFunction:
        name = self.name or f.__name__
        return LabelingFunction(
            name=name,
            f=f,
            label_space=self.label_space,
            schema=self.schema,
            resources=self.resources,
            preprocessors=self.preprocessors,
            fault_tolerant=self.fault_tolerant,
        )
