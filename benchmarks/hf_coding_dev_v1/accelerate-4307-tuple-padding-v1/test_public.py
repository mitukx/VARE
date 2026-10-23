from accelerate import PartialState
def test_tuple_padding():
    state=PartialState()
    state.__dict__["num_processes"]=2
    state.__dict__["process_index"]=1
    values=(0,1,2)
    with state.split_between_processes(values, apply_padding=True) as result:
        assert isinstance(result, tuple)
        assert result == (2,2)
