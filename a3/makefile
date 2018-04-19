CC=pgcc
CFLAGS=-fast -acc -Minfo -ta=tesla,cc35 -O2
LDFLAGS=-acc -ta=tesla,cc35

#make the program
assig3.x: main.c
	$(CC) $(LDFLAGS) -o$@ $?

#cleanup function
clean:
	rm *.x out*.ppm

#-gencode arch=compute_20,code=sm_20
