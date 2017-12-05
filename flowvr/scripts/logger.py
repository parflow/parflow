import matplotlib.pyplot as plt
import sys
import flowvr

# https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib

plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    #min_x = 0
    #max_x = 10

    def __init__(self, title):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.figure.canvas.set_window_title(title)
        self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylabel(title)
        self.ax.set_xlabel('iteration');
        #Other stuff
        self.ax.grid()

    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


datas = {}
updaters = {}
names = sys.argv[1:]
for name in names:
 datas[name] = []
 updaters[name] = DynamicUpdate(name)
 print ("Add name %s" % name)

nextX = 0

xdata = []

ports = flowvr.vectorPort()
port = flowvr.InputPort('in')
ports.push_back(port)

module = flowvr.initModule(ports)
while module.wait():
  message = port.get()
  #print ("will log it!")

  stamps = message.getStamps()

  xdata.append(nextX)
  nextX += 1
  for name in names:
    val = float(stamps[name])
    datas[name].append(val)
    print("%s:%f" % (name, val))
    du = updaters[name]
    du.on_running(xdata, datas[name])

  #print(datas)


module.close()
plt.close()
