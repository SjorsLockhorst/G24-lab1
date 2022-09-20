import abc

from machine_learning import load_model

log_reg = load_model("log_reg.pickle")


class StateInterface(metaclass=abc.ABCMeta):
    def __init__(self, number, next_states, end=False):
        self.number = number
        self.next_states = next_states
        self.end = end

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "activate") and callable(subclass.activate)

    @abc.abstractmethod
    def activate(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


class WelcomeState(StateInterface):
    def __init__(self, number, next_states):
        super().__init__(number, next_states)

    def activate(self):
        print("Welcome user")


class ThankYouState(StateInterface):
    def __init__(self, number, next_states):
        super().__init__(number, next_states)

    def activate(self):
        print("Thanks!")


class ByeState(StateInterface):
    def __init__(self, number, next_states):
        super().__init__(number, next_states, end=True)

    def activate(self):
        print("Good bye")


def transition(
    state: StateInterface, sentence: str, model=log_reg, verbose=False
) -> StateInterface:
    dialog_act = model.predict([sentence.lower()])[0]
    if verbose:
        print(f"Dialog act: {dialog_act}")
        print(f"Previous state: {state}")
        print(f"Next state: 'TODO: put next state'")

    return state.next_states


if __name__ == "__main__":
    # Activate first state
    # While state.end == False
    bye = ByeState(1, None)
    welcome = WelcomeState(1, bye)

    state = welcome
    VERBOSE = True
    while not state.end:
        state.activate()
        user_input = input()
        state = transition(state, user_input, verbose=VERBOSE)
    state.activate()
