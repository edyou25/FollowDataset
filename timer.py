import tkinter as t,time
s=[time.time()]
r=t.Tk(); l=t.Label(r,font=("Arial",40)); l.pack()
r.bind("<space>",lambda e:s.__setitem__(0,time.time()))
def u(): l.config(text=f"{time.time()-s[0]:.1f}s"); r.after(100,u)
u(); r.mainloop()
