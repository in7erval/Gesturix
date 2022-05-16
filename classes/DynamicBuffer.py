class DynamicBuffer:

    def __init__(self, buffer_size=10):
        self.__buffer_size = buffer_size
        self.__current_size = 0
        self.__buffer = []

    def save(self, data):
        if data:
            self.__buffer.append(data)
            self.__current_size += 1
            if len(self.__buffer) > self.__buffer_size:
                self.__buffer.pop(0)
                self.__current_size = self.__buffer_size

    def is_full(self):
        return len(self.__buffer) == self.__buffer_size

    def clear(self):
        self.__buffer = []
        self.__current_size = 0

    def get(self):
        return self.__buffer
