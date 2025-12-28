# Problem: unicode1
(a) '\x00'
(b) "'\\x00'" is the string representation: printed it looks blank?
(c) ah - the character is simply empty space! so it basically does nothing. It is ignored during printing. (Looking online returned that this is the null character!)

# Problem: unicode2
(a) I suspect utf-16 and utf-32 take up more space (16 and 32 bits) in the encoding
(b) the issue is that each character is not necessarily encoded into one single byte. A single char might turn into many bytes. If that happens, we need to decode the bytes at once and not one by one! E.g., an accent character like e with chapeau.
(c) ???

# Problem 3: 
