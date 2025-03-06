def lookup(mapping, key, message="unknown key: {key}"):
    sentinel = object()
    value = mapping.get(key, sentinel)
    if value is sentinel:
        raise ValueError(message.format(key=key))

    return value
