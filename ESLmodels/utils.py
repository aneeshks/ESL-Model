from functools import wraps


def lazy_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        store_name = '_lazy_' + func.__name__
        value = getattr(self, store_name, None)
        if value is None:
            value = func(self, *args, **kwargs)
            setattr(self, store_name, value)
        return value
    return wrapper