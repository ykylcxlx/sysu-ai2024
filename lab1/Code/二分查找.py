def BinarySearch(nums, target):
    low = 0
    high = len(nums) - 1
    if nums == None or nums[0]>target or nums[high]<target:
        return -1
    while(low <= high):
        mid = int(low +(high - low)//2)#注意整除
        if(target < nums[mid]):
            high = mid - 1
        elif(target > nums[mid]):
            low = mid + 1
        else:
            while( mid > low and nums[mid-1]==nums[mid]):
                mid = mid - 1
            return mid
    return -1

 
if __name__ == "__main__":
    nums = list(range(1,100000,2))
    #nums = [1,2,51,51,51,51,78,90]
    target = 51
    print("the index is:",BinarySearch(nums, target))
    target = 10
    print("the index is:",BinarySearch(nums, target))
    """

    :param nums: list[int]

    :param target: int

    :return: int

    """
