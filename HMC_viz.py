import matplotlib
# Needed for blit animation
matplotlib.use('GTKAgg')

from pylab import *
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import itertools,pdb

plt.close('all')

image_format = 'png'

####################################
##### Setup
####################################

def multi_gauss(x,mu,sig):
    """
    Multi-variate Gaussian function (will also take scalar input)
    """
    if type(x) == float or type(x) == float64:
	k=1
	x = array([x]); mu = array([mu]); sig = array([sig])
	icov = 1./sig**2
	dcov = sig**2
    elif type(x)==array or type(x)==list:
	k = len(x)
	x = array(x); mu = array(mu); sig = array(sig)
	assert k == len(mu) == shape(sig)[0] == shape(sig)[1]
	icov = inv(sig**2)
	dcov = det(sig**2)
    else:
	raise Exception("Pass float or array to multi_gauss")
    return (2*pi)**(-k/2.) * dcov**-.5 * exp(dot(-.5*(x-mu).T , dot(icov, (x-mu))))

def v_multi_gauss(x,mu,sig):
    vmg = vectorize(lambda xp: multi_gauss(xp,mu,sig))
    return vmg(x)

def v2_multi_gauss(x,mu,sig):
    zs = np.zeros(shape(x[0]))
    for i in range(shape(x[0])[0]):
	for j in range(shape(x[0])[1]):
	    zs[i,j] = multi_gauss([x[0][i,j],x[1][i,j]],mu,sig)
    return zs



####################################
##### 1D function
####################################

xs = linspace(-3,3,11)
ys = v_multi_gauss(xs,0,1)
xs2 = linspace(-3,3,1000)
ys2 = v_multi_gauss(xs2,0,1)


fig,ax = plt.subplots(1)
fig.patch.set_alpha(0.)
plt.axis([min(xs),min(xs),min(ys),max(ys)*1.1])
plt.xticks(xs)
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
ax.grid(b=1,axis='x')
ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.1f'))


plt.savefig('HMC_1D_grid.'+image_format)

ax.plot(xs,ys,'o',c='k',zorder=1,ms=10)

plt.savefig('HMC_1D_points.'+image_format)

ax.plot(xs2,ys2,mec='none',c='0.5',zorder=0,lw=2)

plt.savefig('HMC_1D_line.'+image_format)




####################################
##### 2D function
####################################

xs = linspace(-3,3,11)
ys = linspace(-3,3,11)
xx,yy = np.meshgrid(xs,ys)
xys = array([i for i in itertools.product(xs,xs)])
zs = v2_multi_gauss([xx,yy],[0,0],[[1,0],[0,1]])
xs2 = linspace(-3,3,300)
ys2 = linspace(-3,3,300)
xx2,yy2 = np.meshgrid(xs2,ys2)
zs2 = v2_multi_gauss([xx2,yy2],[0,0],[[1,0],[0,1]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_alpha(0.)
ax.set_xlim3d(min(xs),max(xs))
ax.set_ylim3d(min(ys),max(ys))
ax.set_zlim3d(0,np.max(zs)*1.1)
#plt.xticks(xs[::2]); plt.yticks(ys[::2])
plt.xlabel('$x$')
plt.ylabel('$y$')
ax.set_zlabel('$f(x,y)$')
ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.1f'))

#ax.grid(b=1,axis='x')
plt.savefig('HMC_2D_grid.'+image_format)

ax.scatter(xx.ravel(),yy.ravel(),zs.ravel(),'o',edgecolor='none',c='k',zorder=1,s=20)

plt.savefig('HMC_2D_points.'+image_format)

ax.plot_surface(xx2,yy2,zs2,edgecolor='none',color='0.5',zorder=0,alpha=0.5)

plt.savefig('HMC_2D_line.'+image_format)




####################################
##### Simulate ball
####################################

def hamiltonian_sim(fsurf,x0,p0,m=1.,dt=.5e-3,Nsteps=10,sdx=0.001):
    """
    Simulate Hamiltonian dynamics given a function specifying a surface in K dimensions (fsurf),
    an initial position (x0) and momentum (p0), mass (m), a timestep (dt), and a trajectory length (Nsteps)
    
    sdx is the fixed step size over which to estimate the potential gradient 
    
    This uses the modified Eulerian discretization, rather than the (more accurate) leapfrog
    """
    # Initialize trajectory arrays
    K = len(x0)
    x = np.zeros([Nsteps,K]); p = np.zeros([Nsteps,K])
    PE = np.zeros([Nsteps])
    x[0] = x0; p[0] = p0
    dHdx = np.zeros(K)
    # Calculate initial potential energy
    PE[0] = m * fsurf(x[0])
    # Step through the simulation
    for n in range(1,Nsteps):
	# Update position
	x[n] = x[n-1] + p[n-1] / m * dt
	# Update potential energy
	PE[n] = m * fsurf(x[n])
	# Estimate potential gradient
	for k in range(K):
	    dx = np.zeros(K); dx[k] = sdx
	    dHdx[k] = m * (fsurf(x[n] + dx) - fsurf(x[n])) / sdx
	# Update momentum
	p[n] = p[n-1] - (2 * dHdx / m) * dt
	if isnan(p[n][0]): pdb.set_trace()
    KE = 0.5 * m * norm(p,axis=1)**2
    return x,p,PE,KE

pfunc = lambda x: -log(multi_gauss([x[0],x[1]],[0,0],[[1,0],[0,1]]))
p_var = 1.

## Calculate bigger 2D gaussian surface
xs2 = linspace(-10,10,1000)
ys2 = linspace(-10,10,1000)
xx2,yy2 = np.meshgrid(xs2,ys2)
zs2 = v2_multi_gauss([xx2,yy2],[0,0],[[1,0],[0,1]])


### Simulate and plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_alpha(0.)
ax.set_xlim3d(min(xs2),max(xs2))
ax.set_ylim3d(min(ys2),max(ys2))
ax.set_zlim3d(-np.max(zs)*1.1,0)
#plt.xticks(xs[::2]); plt.yticks(ys[::2])
plt.xlabel('$x$')
plt.ylabel('$y$')
ax.set_zlabel('$f(x,y)$')
ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.1f'))

ax.plot_surface(xx2[::3,::3],yy2[::3,::3],-zs2[::3,::3],edgecolor='none',color='0.5',zorder=0,alpha=0.5)

plt.savefig('HMC_sim_-1.'+image_format)


x0 = [9,9]
p0 = normal(0,p_var,[2])
Nsim = 20; Nsteps = 50
all_x = np.zeros([Nsim * Nsteps,2])
all_PE = np.zeros([Nsim * Nsteps])
for i in range(Nsim):
    col = cm.Dark2(i/float(Nsim))
    ax.scatter(x0[0], x0[1], -exp(-pfunc(x0)),color=col)
    sim_x,sim_p,sim_PE,sim_KE = hamiltonian_sim(pfunc,x0,p0,dt=3e-2,Nsteps=Nsteps)
    ax.plot(sim_x[:,0], sim_x[:,1], -exp(-sim_PE), lw=2, color=col,alpha=0.3)
    x0 = sim_x[-1]
    #p0 = sim_p[-1] # for continuous trajectory
    p0 = normal(0,p_var,[2]) # for sampled trajectory - using a value << mass to set the variance, since I don't have a metropolis step to reject bad moves
    plt.savefig('HMC_sim_'+str(i)+'.'+image_format)
    # Save for future use
    all_x[i*Nsteps:(i+1)*Nsteps] = sim_x
    all_PE[i*Nsteps:(i+1)*Nsteps] = sim_PE

ax.view_init(elev=90., azim=0.)
ax.set_zticks([])
ax.set_zlabel('')
plt.savefig('HMC_sim_top.'+image_format)


####################################
##### Animated version
####################################

global x0,p0,cline,gsurf
## Initial parameters
x0 = [-1.,-1.]
p0 = normal(0,p_var,[2])
Nsim = 10
start_zoom = 100
end_zoom = 200
zoom_fact = 3

## Setup figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_alpha(0.)
plt.xticks([]); plt.yticks([]); ax.set_zticks([]) # If you have labels, they mess up the border of the animation
plt.xlabel('')
plt.ylabel('')
ax.get_xaxis().set_major_formatter(FormatStrFormatter('%0.1f'))
## Plot the initial surface
gsurf = ax.plot_surface(xx2[::int(zoom_fact**2)],yy2[::int(zoom_fact**2)],-zs2[::int(zoom_fact**2)],edgecolor='none',color='0.5',zorder=0,alpha=0.2)

def init():
    ax.set_xlim3d(min(xs2),max(xs2))
    ax.set_ylim3d(min(ys2),max(ys2))
    ax.set_zlim3d(-np.max(zs)*1.1,0)
    ax.view_init(elev=30., azim=0.)
    return ax,

def animate(i):
    global x0,p0,cline,gsurf
    col = cm.Dark2((i/Nsteps)/float(Nsim))
    # Shift view
    ax.view_init(elev=30., azim=i/2.)
    if i > start_zoom and i < end_zoom:
	zf = 1+(i-start_zoom)/float(end_zoom-start_zoom)*(zoom_fact-1)
	sf = int((zoom_fact / zf))
	bounds = [min(xs2)/zf,max(xs2)/zf,min(ys2)/zf,max(ys2)/zf]
	i_bounds = [int(shape(xx2)[0]*(zf-1)/2.),int(shape(xx2)[0]/(zf-((zf-1)/2))),int(shape(xx2)[1]*(zf-1)/2.),int(shape(xx2)[1]/(zf-((zf-1)/2)))]
	ax.set_xlim3d(bounds[0:2])
	ax.set_ylim3d(bounds[2:])
	## Replot the surface within this area
	gsurf.remove()
	## This does not work because sel flattens the array and its not trivial to ravel
	sel = where((xx2 > bounds[0]) & (xx2 < bounds[1]) & (yy2 > bounds[2]) & (yy2 < bounds[3]))
	pa = shape(xx2[sel])[0]**.5
	xs3 = xx2[sel].reshape([pa,pa])
	ys3 = yy2[sel].reshape([pa,pa])
	zs3 = zs2[sel].reshape([pa,pa])
	gsurf = ax.plot_surface(xs3,ys3,-zs3,edgecolor='none',color='0.5',zorder=0,alpha=0.2)
    # Remove old trajectory
    try:
	cline[0].remove()
    except:
	print "starting new line"
    # Iterate particle
    left = i/Nsteps * Nsteps
    cline = ax.plot(all_x[left:i,0], all_x[left:i,1], -exp(-all_PE[left:i]), lw=2, color=col)
    # If this is the end of the line, hide cline
    if mod(i+1,Nsteps) == 0:
	del cline
	ax.scatter(all_x[i,0], all_x[i,1], -exp(-pfunc(all_x[i])), color=col)
    draw()
    return ax,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=Nsim*Nsteps*2, interval=1, blit=True) # Not sure why you need the *2, but it seems to stop halfway without it

# For some reason, matplotlib corrupts the output...  but I found that I can fix it using VLC > Media > Convert / Save
anim.save('HMC_sim.mp4', writer='ffmpeg', fps=30, bitrate = 2048)  # need matplotlib 1.3

