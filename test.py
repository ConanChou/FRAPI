str1 = "aaaaabbaacccaaaaaaaa"
str2 = "bbbbaab"

# longest_run(str1) = 5
# longest_run(str2) = 4

def longest_run(str):
    longest = 0
    counter = 1
    i, j = 0, 0

    for j in range(len(str)-1):
        if str[i] == str[j+1]:
            counter += 1
        else:
            if longest < counter:
                longest = counter
            counter = 1
            i = j
    return longest if longest > counter else counter + 1

if __name__ == '__main__':
    print longest_run(str1)
    print longest_run(str2)


