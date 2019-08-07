#!/usr/bin/env python

import json
import os
import sys
from importlib import import_module
from typing import Any, List

PACKAGE_INFO_PATH = "docs/packages.json"
PACKAGE_PAGE_PATH = "docs/packages"


PACKAGE_DOC_TEMPLATE = """{title}
{underscore}

{docstring}

.. currentmodule:: snorkel.{package_name}

.. autosummary::
   :toctree: _autosummary/{package_name}/
   :nosignatures:

   {members}
"""


def get_title_and_underscore(package_name: str) -> str:
    title = f"Snorkel {package_name.capitalize()} Package"
    underscore = "-" * len(title)
    return title, underscore


def get_package_members(package: Any) -> List[str]:
    members = []
    for name in dir(package):
        if name.startswith("_"):
            continue
        obj = getattr(package, name)
        if isinstance(obj, type) or callable(obj):
            members.append(name)
    return members


def main(check: bool) -> None:
    with open(PACKAGE_INFO_PATH, "r") as f:
        packages_info = json.load(f)
    package_names = sorted(packages_info["packages"])
    if check:
        f_basenames = sorted(
            [
                os.path.splitext(f_name)[0]
                for f_name in os.listdir(PACKAGE_PAGE_PATH)
                if f_name.endswith(".rst")
            ]
        )
        if f_basenames != package_names:
            raise ValueError(
                "Expected package files do not match actual!\n"
                f"Expected: {package_names}\n"
                f"Actual: {f_basenames}"
            )
    else:
        os.makedirs(PACKAGE_PAGE_PATH, exist_ok=True)
    for package_name in package_names:
        package = import_module(f"snorkel.{package_name}")
        docstring = package.__doc__
        title, underscore = get_title_and_underscore(package_name)
        all_members = get_package_members(package)
        all_members.extend(packages_info["extra_members"].get(package_name, []))
        contents = PACKAGE_DOC_TEMPLATE.format(
            title=title,
            underscore=underscore,
            docstring=docstring,
            package_name=package_name,
            members="\n   ".join(sorted(all_members, key=lambda s: s.split(".")[-1])),
        )
        f_path = os.path.join(PACKAGE_PAGE_PATH, f"{package_name}.rst")
        if check:
            with open(f_path, "r") as f:
                contents_actual = f.read()
            if contents != contents_actual:
                raise ValueError(f"Contents for {package_name} differ!")
        else:
            with open(f_path, "w") as f:
                f.write(contents)


if __name__ == "__main__":
    check = False if len(sys.argv) == 1 else (sys.argv[1] == "--check")
    sys.exit(main(check))
