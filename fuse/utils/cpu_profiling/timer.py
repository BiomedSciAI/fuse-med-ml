import time


class Timer:
    def __init__(self, txt: str = ""):
        self.txt = txt

    def __enter__(self) -> None:
        if len(self.txt) > 0:
            print("starting Timer for ", self.txt)
        self.start = time.time()

    def __exit__(self, *args: list) -> None:
        self.end = time.time()
        print(
            f"{self.txt} took {self.end-self.start} seconds."
        )  # todo: use minutes, hours, etc. depending on the value


if __name__ == "__main__":

    def foo(x):  # type: ignore
        with Timer("banana"):
            for _ in range(10**5):
                x = x * 2

    foo(20)
