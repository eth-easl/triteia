from nvidia import nvcomp

_SUPPORTED_ALGORITHMS = ["gdeflate:ht"]


class LosslessCompressor:
    def __init__(self, algorithm="gdeflate:ht"):
        if algorithm == "gdeflate:ht":
            self.codec = nvcomp.Codec(algorithm="GDeflate", algorithm_type=0)
        else:
            raise ValueError(
                f"Unsupported algorithm. Expect one of {_SUPPORTED_ALGORITHMS}, got {algorithm}"
            )
