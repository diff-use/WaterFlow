import random
from typing import Dict, Tuple, Callable, Any, Sequence, Union, Mapping, Optional, List, Set
from collections import OrderedDict
import inspect
import logging
from copy import deepcopy

import torch
import torch.jit

from e3nn import o3


from torch_geometric.data import Batch

from SLAE.util import _fix_irreps_dict, _irreps_compatible




_GLOBAL_ALL_ASKED_FOR_KEYS: Set[str] = set()
class Config:
    """
    Minimal param-collector used by `instantiate(...)`.
    Supports:
      - from_class / from_function: seed defaults from a builder signature
      - allow-listing when kwargs are removed
      - update(...) and update_w_prefix(...) to collect args
      - as_dict() to expose the final key->value map
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        allow_list: Optional[List[str]] = None,
    ):
        self._items: Dict[str, Any] = {}
        self._allow_list: List[str] = allow_list or []
        self._allow_all: bool = allow_list is None  # allow any key if no allow_list
        if config:
            self.update(config)

    # --- basics ---
    def allow_list(self) -> List[str]:
        return list(self._allow_list)

    @staticmethod
    def as_dict(obj: "Config") -> Dict[str, Any]:
        return obj._items.copy()

    # --- setters ---
    def _allowed(self, key: str) -> bool:
        return self._allow_all or (key in self._allow_list)

    def update(self, dictionary: dict):
        """
        Insert key/value pairs that are allowed.
        Returns the set of keys actually written.
        """
        written = set()
        for k, v in dictionary.items():
            if not self._allowed(k):
                continue
            self._items[k] = deepcopy(v)
            written.add(k)
        return written

    def update_w_prefix(self, dictionary: dict, prefix: str):
        """
        Accept keys that look like  f'{prefix}_{name}'  and write them as  'name'.
        Returns a mapping of logical-key -> source-key for logging upstream.
        """
        written_map = {}
        head = prefix + "_"
        L = len(head)
        for src_k, v in dictionary.items():
            if not src_k.startswith(head):
                continue
            k = src_k[L:]
            if not self._allowed(k):
                continue
            self._items[k] = deepcopy(v)
            written_map[k] = src_k
        # Special case: pass through nested kwargs if present (caller handles merging)
        # We don't transform them here; we only let the caller read `{name}_kwargs` later.
        return written_map

    # --- constructors from callables ---
    @staticmethod
    def from_function(function, remove_kwargs: bool = False) -> "Config":
        """
        Build a config seeded with defaults from a callable's signature.
        If remove_kwargs=True, only named parameters are allowed (no **kwargs).
        Otherwise, config accepts any keys (allow_all).
        """
        sig = inspect.signature(function)
        # defaults
        defaults = {
            k: p.default
            for k, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        # parameter names (skip self for __init__)
        param_keys = [k for k in sig.parameters.keys() if k != "self"]

        if "kwargs" in param_keys and not remove_kwargs:
            # accept any keys
            return Config(config=defaults, allow_list=None)
        else:
            # restrict to explicit params (minus kwargs if removed)
            if "kwargs" in param_keys:
                param_keys.remove("kwargs")
            return Config(config=defaults, allow_list=param_keys)

    @staticmethod
    def from_class(class_type, remove_kwargs: bool = False) -> "Config":
        if inspect.isclass(class_type):
            return Config.from_function(class_type.__init__, remove_kwargs)
        elif callable(class_type):
            return Config.from_function(class_type, remove_kwargs)
        else:
            raise ValueError("from_class expects a class or callable")


def instantiate(
    builder,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: dict = None,
    all_args: dict = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
    parent_builders: list = [],
):
    """Automatic initializing class instance by matching keys in the parameter dictionary to the constructor function.

    Keys that are exactly the same, or with a 'prefix_' in all_args, optional_args will be used.
    Priority:

        all_args[key] < all_args[prefix_key] < optional_args[key] < optional_args[prefix_key] < positional_args

    Args:
        builder: the type of the instance
        prefix: the prefix used to address the parameter keys
        positional_args: the arguments used for input. These arguments have the top priority.
        optional_args: the second priority group to search for keys.
        all_args: the third priority group to search for keys.
        remove_kwargs: if True, ignore the kwargs argument in the init funciton
            same definition as the one in Config.from_function
        return_args_only (bool): if True, do not instantiate, only return the arguments
    """

    prefix_list = [builder.__name__] if inspect.isclass(builder) else []
    if isinstance(prefix, str):
        prefix_list += [prefix]
    elif isinstance(prefix, list):
        prefix_list += prefix
    else:
        raise ValueError(f"prefix has the wrong type {type(prefix)}")

    # detect the input parameters needed from params
    config = Config.from_class(builder, remove_kwargs=remove_kwargs)

    # be strict about _kwargs keys:
    allow = config.allow_list()
    for key in allow:
        bname = key[:-7]
        if key.endswith("_kwargs") and bname not in allow:
            raise KeyError(
                f"Instantiating {builder.__name__}: found kwargs argument `{key}`, but no parameter `{bname}` for the corresponding builder. (Did you rename `{bname}` but forget to change `{bname}_kwargs`?) Either add a parameter for `{bname}` if you are trying to allow construction of a submodule, or, if `{bname}_kwargs` is just supposed to be a dictionary, rename it without `_kwargs`."
            )
    del allow

    key_mapping = {}
    if all_args is not None:
        # fetch paratemeters that directly match the name
        _keys = config.update(all_args)
        key_mapping["all"] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                all_args,
                prefix=prefix_str,
            )
            key_mapping["all"].update(_keys)

    if optional_args is not None:
        # fetch paratemeters that directly match the name
        _keys = config.update(optional_args)
        key_mapping["optional"] = {k: k for k in _keys}
        # fetch paratemeters that match prefix + "_" + name
        for idx, prefix_str in enumerate(prefix_list):
            _keys = config.update_w_prefix(
                optional_args,
                prefix=prefix_str,
            )
            key_mapping["optional"].update(_keys)

    # for logging only, remove the overlapped keys
    if "all" in key_mapping and "optional" in key_mapping:
        key_mapping["all"] = {
            k: v
            for k, v in key_mapping["all"].items()
            if k not in key_mapping["optional"]
        }

    final_optional_args = Config.as_dict(config)

    # for nested argument, it is possible that the positional args contain unnecesary keys
    if len(parent_builders) > 0:
        _positional_args = {
            k: v for k, v in positional_args.items() if k in config.allow_list()
        }
        positional_args = _positional_args

    init_args = final_optional_args.copy()
    init_args.update(positional_args)

    # find out argument for the nested keyword
    search_keys = [key for key in init_args if key + "_kwargs" in config.allow_list()]
    for key in search_keys:
        sub_builder = init_args[key]
        if sub_builder is None:
            # if the builder is None, skip it
            continue

        if not (callable(sub_builder) or inspect.isclass(sub_builder)):
            raise ValueError(
                f"Builder for submodule `{key}` must be a callable or a class, got `{sub_builder!r}` instead."
            )

        # add double check to avoid cycle
        # only overwrite the optional argument, not the positional ones
        if (
            sub_builder not in parent_builders
            and key + "_kwargs" not in positional_args
        ):
            sub_prefix_list = [sub_builder.__name__, key]
            for prefix in prefix_list:
                sub_prefix_list = sub_prefix_list + [
                    prefix,
                    prefix + "_" + key,
                ]

            nested_km, nested_kwargs = instantiate(
                sub_builder,
                prefix=sub_prefix_list,
                positional_args=positional_args,
                optional_args=optional_args,
                all_args=all_args,
                remove_kwargs=remove_kwargs,
                return_args_only=True,
                parent_builders=[builder] + parent_builders,
            )
            # the values in kwargs get higher priority
            nested_kwargs.update(final_optional_args.get(key + "_kwargs", {}))
            final_optional_args[key + "_kwargs"] = nested_kwargs

            for t in key_mapping:
                key_mapping[t].update(
                    {key + "_kwargs." + k: v for k, v in nested_km[t].items()}
                )
        elif sub_builder in parent_builders:
            raise RuntimeError(
                f"cyclic recursion in builder {parent_builders} {sub_builder}"
            )
        elif not callable(sub_builder) and not inspect.isclass(sub_builder):
            logging.warning(f"subbuilder is not callable {sub_builder}")
        elif key + "_kwargs" in positional_args:
            logging.warning(
                f"skip searching for nested argument because {key}_kwargs are defined in positional arguments"
            )

    # remove duplicates
    for key in positional_args:
        final_optional_args.pop(key, None)
        for t in key_mapping:
            key_mapping[t].pop(key, None)

    # debug info
    if len(parent_builders) == 0:
        # ^ we only want to log or consume arguments for the "unused keys" check
        #   if this is a root-level build. For subbuilders, we don't want to log
        #   or, worse, mark keys without prefixes as consumed.
        logging.debug(
            f"{'get args for' if return_args_only else 'instantiate'} {builder.__name__}"
        )
        for t in key_mapping:
            for k, v in key_mapping[t].items():
                string = f" {t:>10s}_args :  {k:>50s}"
                # key mapping tells us how values got from the
                # users config (v) to the object being built (k)
                # thus v is by definition a valid key
                _GLOBAL_ALL_ASKED_FOR_KEYS.add(v)
                if k != v:
                    string += f" <- {v:>50s}"
                logging.debug(string)
        logging.debug(f"...{builder.__name__}_param = dict(")
        logging.debug(f"...   optional_args = {final_optional_args},")
        logging.debug(f"...   positional_args = {positional_args})")

    # Short circuit for return_args_only
    if return_args_only:
        return key_mapping, final_optional_args
    # Otherwise, actually build the thing:
    try:
        instance = builder(**positional_args, **final_optional_args)
    except Exception as e:
        raise RuntimeError(
            f"Failed to build object with prefix `{prefix}` using builder `{builder.__name__}`"
        ) from e

    return instance, final_optional_args


class GraphModuleMixin:
    r"""Mixin parent class for ``torch.nn.Module``s that act on and return Batch graph data.

    All such classes should call ``_init_irreps`` in their ``__init__`` functions with information on the data 
    fields they expect, require, and produce, as well as their corresponding irreps.
    """

    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        my_irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        irreps_out: Dict[str, Any] = {},
    ):
        """Setup the expected data fields and their irreps for this graph module.

        ``None`` is a valid irreps in the context for anything that is invariant but not well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph, which are invariant but are integers, not ``0e`` scalars.

        Args:
            irreps_in (dict): maps names of all input fields from previous modules or
                data to their corresponding irreps
            my_irreps_in (dict): maps names of fields to the irreps they must have for
                this graph module. Will be checked for consistancy with ``irreps_in``
            required_irreps_in: sequence of names of fields that must be present in
                ``irreps_in``, but that can have any irreps.
            irreps_out (dict): mapping names of fields that are modified/output by
                this graph module to their irreps.
        """
        # Coerce
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = _fix_irreps_dict(irreps_in)
        # positions are *always* 1o, and always present
        if "pos" in irreps_in:
            if irreps_in["pos"] != o3.Irreps("1x1o"):
                raise ValueError(
                    f"Positions must have irreps 1o, got instead `{irreps_in['pos']}`"
                )
        irreps_in["pos"] = o3.Irreps("1o")
        # edges are also always present
        if "edge_index" in irreps_in:
            if irreps_in["edge_index"] is not None:
                raise ValueError(
                    f"Edge indexes must have irreps None, got instead `{irreps_in['edge_index']}`"
                )
        irreps_in["edge_index"] = None

        my_irreps_in = _fix_irreps_dict(my_irreps_in)

        irreps_out = _fix_irreps_dict(irreps_out)
        # Confirm compatibility:
        # with my_irreps_in
        for k in my_irreps_in:
            if k in irreps_in and irreps_in[k] != my_irreps_in[k]:
                raise ValueError(
                    f"The given input irreps {irreps_in[k]} for field '{k}' is incompatible with this configuration {type(self)}; should have been {my_irreps_in[k]}"
                )
        # with required_irreps_in
        for k in required_irreps_in:
            if k not in irreps_in:
                raise ValueError(
                    f"This {type(self)} requires field '{k}' to be in irreps_in"
                )
        # Save stuff
        self.irreps_in = irreps_in
        # The output irreps of any graph module are whatever inputs it has, overwritten with whatever outputs it has.
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out

    def _add_independent_irreps(self, irreps: Dict[str, Any]):
        """
        Insert some independent irreps that need to be exposed to the self.irreps_in and self.irreps_out.
        The terms that have already appeared in the irreps_in will be removed.

        Args:
            irreps (dict): maps names of all new fields
        """

        irreps = {
            key: irrep for key, irrep in irreps.items() if key not in self.irreps_in
        }
        irreps_in = _fix_irreps_dict(irreps)
        irreps_out = _fix_irreps_dict(
            {key: irrep for key, irrep in irreps.items() if key not in self.irreps_out}
        )
        self.irreps_in.update(irreps_in)
        self.irreps_out.update(irreps_out)

    def _make_tracing_inputs(self, n):
        # We impliment this to be able to trace graph modules
        out = []
        for _ in range(n):
            batch = random.randint(1, 4)
            # TODO: handle None case
            # TODO: do only required inputs
            # TODO: dummy input if empty?
            out.append(
                {
                    "forward": (
                        {
                            k: i.randn(batch, -1)
                            for k, i in self.irreps_in.items()
                            if i is not None
                        },
                    )
                }
            )
        return out


class SequentialGraphNetwork(GraphModuleMixin, torch.nn.Sequential):
    r"""A ``torch.nn.Sequential`` of ``GraphModuleMixin``s.

    Args:
        modules (list or dict of ``GraphModuleMixin``s): the sequence of graph modules. If a list, the modules will be named ``"module0", "module1", ...``.
    """

    def __init__(
        self,
        modules: Union[Sequence[GraphModuleMixin], Dict[str, GraphModuleMixin]],
    ):
        if isinstance(modules, dict):
            module_list = list(modules.values())
        else:
            module_list = list(modules)
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:]):
            assert _irreps_compatible(
                m1.irreps_out, m2.irreps_in
            ), f"Incompatible irreps_out from {type(m1).__name__} for input to {type(m2).__name__}: {m1.irreps_out} -> {m2.irreps_in}"
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        # torch.nn.Sequential will name children correctly if passed an OrderedDict
        if isinstance(modules, dict):
            modules = OrderedDict(modules)
        else:
            modules = OrderedDict((f"module{i}", m) for i, m in enumerate(module_list))
        super().__init__(modules)

    @classmethod
    def from_parameters(
        cls,
        shared_params: Mapping,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
        irreps_in: Optional[dict] = None,
    ):
        r"""Construct a ``SequentialGraphModule`` of modules built from a shared set of parameters.

        For some layer, a parameter with name ``param`` will be taken, in order of priority, from:
          1. The specific value in the parameter dictionary for that layer, if provided
          2. ``name_param`` in ``shared_params`` where ``name`` is the name of the layer
          3. ``param`` in ``shared_params``

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            layers (dict): dictionary mapping unique names of layers to either:
                  1. A callable (such as a class or function) that can be used to ``instantiate`` a module for that layer
                  2. A tuple of such a callable and a dictionary mapping parameter names to values. The given dictionary of parameters will override for this layer values found in ``shared_params``.
                Options 1. and 2. can be mixed.
            irreps_in (optional dict): ``irreps_in`` for the first module in the sequence.

        Returns:
            The constructed SequentialGraphNetwork.
        """
        # note that dictionary ordered gueranteed in >=3.7, so its fine to do an ordered sequential as a dict.
        built_modules = []
        for name, builder in layers.items():
            if not isinstance(name, str):
                raise ValueError(f"`'name'` must be a str; got `{name}`")
            if isinstance(builder, tuple):
                builder, params = builder
            else:
                params = {}
            if not callable(builder):
                raise TypeError(
                    f"The builder has to be a class or a function. got {type(builder)}"
                )

            instance, _ = instantiate(
                builder=builder,
                prefix=name,
                positional_args=(
                    dict(
                        irreps_in=(
                            built_modules[-1].irreps_out
                            if len(built_modules) > 0
                            else irreps_in
                        )
                    )
                ),
                optional_args=params,
                all_args=shared_params,
            )

            if not isinstance(instance, GraphModuleMixin):
                raise TypeError(
                    f"Builder `{builder}` for layer with name `{name}` did not return a GraphModuleMixin, instead got a {type(instance).__name__}"
                )

            built_modules.append(instance)

        return cls(
            OrderedDict(zip(layers.keys(), built_modules)),
        )

    @torch.jit.unused
    def append(self, name: str, module: GraphModuleMixin) -> None:
        r"""Append a module to the SequentialGraphNetwork.

        Args:
            name (str): the name for the module
            module (GraphModuleMixin): the module to append
        """
        assert _irreps_compatible(self.irreps_out, module.irreps_in)
        self.add_module(name, module)
        self.irreps_out = dict(module.irreps_out)
        return

    @torch.jit.unused
    def append_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and append it.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``

        Returns:
            the build module
        """
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[-1].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.append(name, instance)
        return instance

    @torch.jit.unused
    def insert(
        self,
        name: str,
        module: GraphModuleMixin,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> None:
        """Insert a module after the module with name ``after``.

        Args:
            name: the name of the module to insert
            module: the moldule to insert
            after: the module to insert after
            before: the module to insert before
        """

        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before

        # This checks names, etc.
        self.add_module(name, module)
        # Now insert in the right place by overwriting
        names = list(self._modules.keys())
        modules = list(self._modules.values())
        idx = names.index(insert_location)
        if before is None:
            idx += 1
        names.insert(idx, name)
        modules.insert(idx, module)

        self._modules = OrderedDict(zip(names, modules))

        module_list = list(self._modules.values())

        # sanity check the compatibility
        if idx > 0:
            assert _irreps_compatible(
                module_list[idx - 1].irreps_out, module.irreps_in
            )
        if len(module_list) > idx:
            assert _irreps_compatible(
                module_list[idx + 1].irreps_in, module.irreps_out
            )

        # insert the new irreps_out to the later modules
        for module_id, next_module in enumerate(module_list[idx + 1 :]):
            next_module._add_independent_irreps(module.irreps_out)

        # update the final wrapper irreps_out
        self.irreps_out = dict(module_list[-1].irreps_out)

        return

    @torch.jit.unused
    def insert_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and insert it after ``after``.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``
            after: the name of the module to insert after
            before: the name of the module to insert before

        Returns:
            the inserted module
        """
        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before
        idx = list(self._modules.keys()).index(insert_location) - 1
        if before is None:
            idx += 1
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[idx].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.insert(after=after, before=before, name=name, module=instance)
        return instance

    # Copied from https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
    # with type annotations added
    def forward(self, input: Batch) -> Batch:
        for module in self:
            input = module(input)
        return input
    



class SequentialGraphNetwork(GraphModuleMixin, torch.nn.Sequential):
    r"""A ``torch.nn.Sequential`` of ``GraphModuleMixin``s.

    Args:
        modules (list or dict of ``GraphModuleMixin``s): the sequence of graph modules. If a list, the modules will be named ``"module0", "module1", ...``.
    """

    def __init__(
        self,
        modules: Union[Sequence[GraphModuleMixin], Dict[str, GraphModuleMixin]],
    ):
        if isinstance(modules, dict):
            module_list = list(modules.values())
        else:
            module_list = list(modules)
        # check in/out irreps compatible
        for m1, m2 in zip(module_list, module_list[1:]):
            assert _irreps_compatible(
                m1.irreps_out, m2.irreps_in
            ), f"Incompatible irreps_out from {type(m1).__name__} for input to {type(m2).__name__}: {m1.irreps_out} -> {m2.irreps_in}"
        self._init_irreps(
            irreps_in=module_list[0].irreps_in,
            my_irreps_in=module_list[0].irreps_in,
            irreps_out=module_list[-1].irreps_out,
        )
        # torch.nn.Sequential will name children correctly if passed an OrderedDict
        if isinstance(modules, dict):
            modules = OrderedDict(modules)
        else:
            modules = OrderedDict((f"module{i}", m) for i, m in enumerate(module_list))
        super().__init__(modules)

    @classmethod
    def from_parameters(
        cls,
        shared_params: Mapping,
        layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
        irreps_in: Optional[dict] = None,
    ):
        r"""Construct a ``SequentialGraphModule`` of modules built from a shared set of parameters.

        For some layer, a parameter with name ``param`` will be taken, in order of priority, from:
          1. The specific value in the parameter dictionary for that layer, if provided
          2. ``name_param`` in ``shared_params`` where ``name`` is the name of the layer
          3. ``param`` in ``shared_params``

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            layers (dict): dictionary mapping unique names of layers to either:
                  1. A callable (such as a class or function) that can be used to ``instantiate`` a module for that layer
                  2. A tuple of such a callable and a dictionary mapping parameter names to values. The given dictionary of parameters will override for this layer values found in ``shared_params``.
                Options 1. and 2. can be mixed.
            irreps_in (optional dict): ``irreps_in`` for the first module in the sequence.

        Returns:
            The constructed SequentialGraphNetwork.
        """
        # note that dictionary ordered gueranteed in >=3.7, so its fine to do an ordered sequential as a dict.
        built_modules = []
        for name, builder in layers.items():
            if not isinstance(name, str):
                raise ValueError(f"`'name'` must be a str; got `{name}`")
            if isinstance(builder, tuple):
                builder, params = builder
            else:
                params = {}
            if not callable(builder):
                raise TypeError(
                    f"The builder has to be a class or a function. got {type(builder)}"
                )

            instance, _ = instantiate(
                builder=builder,
                prefix=name,
                positional_args=(
                    dict(
                        irreps_in=(
                            built_modules[-1].irreps_out
                            if len(built_modules) > 0
                            else irreps_in
                        )
                    )
                ),
                optional_args=params,
                all_args=shared_params,
            )

            if not isinstance(instance, GraphModuleMixin):
                raise TypeError(
                    f"Builder `{builder}` for layer with name `{name}` did not return a GraphModuleMixin, instead got a {type(instance).__name__}"
                )

            built_modules.append(instance)

        return cls(
            OrderedDict(zip(layers.keys(), built_modules)),
        )

    @torch.jit.unused
    def append(self, name: str, module: GraphModuleMixin) -> None:
        r"""Append a module to the SequentialGraphNetwork.

        Args:
            name (str): the name for the module
            module (GraphModuleMixin): the module to append
        """
        assert _irreps_compatible(self.irreps_out, module.irreps_in)
        self.add_module(name, module)
        self.irreps_out = dict(module.irreps_out)
        return

    @torch.jit.unused
    def append_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and append it.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``

        Returns:
            the build module
        """
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[-1].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.append(name, instance)
        return instance

    @torch.jit.unused
    def insert(
        self,
        name: str,
        module: GraphModuleMixin,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> None:
        """Insert a module after the module with name ``after``.

        Args:
            name: the name of the module to insert
            module: the moldule to insert
            after: the module to insert after
            before: the module to insert before
        """

        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before

        # This checks names, etc.
        self.add_module(name, module)
        # Now insert in the right place by overwriting
        names = list(self._modules.keys())
        modules = list(self._modules.values())
        idx = names.index(insert_location)
        if before is None:
            idx += 1
        names.insert(idx, name)
        modules.insert(idx, module)

        self._modules = OrderedDict(zip(names, modules))

        module_list = list(self._modules.values())

        # sanity check the compatibility
        if idx > 0:
            assert _irreps_compatible(
                module_list[idx - 1].irreps_out, module.irreps_in
            )
        if len(module_list) > idx:
            assert _irreps_compatible(
                module_list[idx + 1].irreps_in, module.irreps_out
            )

        # insert the new irreps_out to the later modules
        for module_id, next_module in enumerate(module_list[idx + 1 :]):
            next_module._add_independent_irreps(module.irreps_out)

        # update the final wrapper irreps_out
        self.irreps_out = dict(module_list[-1].irreps_out)

        return

    @torch.jit.unused
    def insert_from_parameters(
        self,
        shared_params: Mapping,
        name: str,
        builder: Callable,
        params: Dict[str, Any] = {},
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> GraphModuleMixin:
        r"""Build a module from parameters and insert it after ``after``.

        Args:
            shared_params (dict-like): shared parameters from which to pull when instantiating the module
            name (str): the name for the module
            builder (callable): a class or function to build a module
            params (dict, optional): extra specific parameters for this module that take priority over those in ``shared_params``
            after: the name of the module to insert after
            before: the name of the module to insert before

        Returns:
            the inserted module
        """
        if (before is None) is (after is None):
            raise ValueError("Only one of before or after argument needs to be defined")
        elif before is None:
            insert_location = after
        else:
            insert_location = before
        idx = list(self._modules.keys()).index(insert_location) - 1
        if before is None:
            idx += 1
        instance, _ = instantiate(
            builder=builder,
            prefix=name,
            positional_args=(dict(irreps_in=self[idx].irreps_out)),
            optional_args=params,
            all_args=shared_params,
        )
        self.insert(after=after, before=before, name=name, module=instance)
        return instance


    def forward(self, input: Batch) -> Batch:
        for module in self:
            input = module(input)
        return input