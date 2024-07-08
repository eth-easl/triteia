## Packing and Unpacking 16-bit integers
```c
// packing
int_32 = (int16_1 & 0x0000FFFF) | (int16_2 & 0xFFFF0000);
// unpacking
int16_MSB = (int_32 >> 16) & 0xFFFF;
int16_LSB = int_32 & 0xFFFF;
```