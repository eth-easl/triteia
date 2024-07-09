## Packing and Unpacking 16-bit integers
```c
// packing
int_32 = (int16_1 & 0x0000FFFF) | (int16_2 & 0xFFFF0000);
// unpacking
signed short up  = (meta_ptr->x >> 16) & 0xFFFF;
    signed short low = meta_ptr->x & 0xFFFF;
```