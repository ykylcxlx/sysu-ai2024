def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    #inv = {}
    inv = dict([val,key] for key,val in dict1.items())#括号的位置
    return inv
    
if __name__ == "__main__":
      dict1 ={'Alice':'001', 'Bob':'002'}
      print(ReverseKeyValue(dict1))
