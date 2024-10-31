from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

def semimajor_a(e, c0, m1, m2, G):
    """
    Semimajor axis as a function of eccentricity for a binary
    Peters 1964 eqn. 5.48, a(e)
    :param e: float
        eccentricity of the binary
    :param c0: float
        initial conditions factor
    :param m1: float
        mass_1 of binary
    :param m2: float
        mass_2 of binary
    :param G: float
        gravitational constant
    :return:
        a: float
            semimajor axis distance
    """
    frac = (e**(12.0/19)) / (1 - e**2)
    return c0 * frac * (1 + (121.0/304)*e**2)**(870.0/2299)

def semimajor_a_root(e, a, c0, m1, m2, G):
    """
    Wrapper function for semimajor_a root finding. 
        Ie. rootfinding for an `e` such that semimajor_a(e) = a
        solving for `e` when we have `a`.
    """
    a_new = semimajor_a(e, c0, m1, m2, G)
    return a - a_new

def get_c0(e, a, m1, m2, G):
    """
    Solve for the initial condition which describes the orbit
    :param e: float
        eccentricity
    :param a: float
        semimajor axis distance
    :param m1: float
        mass_1 of binary
    :param m2: float
        mass_2 of binary
    :param G: float
        gravitational constant
    :return:
        c0: float
            initial condition
    """
    # We solve the separation eqn. for a=a0, e=e0
    # see Peters 1964 eqn. 5.48, a(e)
    frac = (e**(12.0/19)) / (1 - e**2)
    c0 = a / (frac * (1 + (121.0/304)*e**2)**(870.0/2299))
    return c0


def period_to_semimajor_a(P, m1, m2, G):
    """
    Convert orital period into semimajor axis distance
    using Kepler's third law
    https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Third_law_of_Kepler
    :param P: float
        orbital period
    :param m1: float
        mass_1 of binary
    :param m2: float
        mass_2 of binary
    :param G: float
        gravitational constant
    :return:
        a: float
            semimajor axis distance
    """
    proportionConst = G*(m1 + m2) / (4*np.pi**2)
    a = (proportionConst * P**2)**(1.0/3)
    return a

def eccentricity_at_shifted_frequency(f, e0, f_ref, m1, m2, G, scalar=False):
    """
    Find the eccentricity at a different reference frequency
    :param f: float
        new reference frequency
    :param e0: float
        old eccentricity value at frequency `f_ref`
    :param f_ref: float
        old reference frequency
    :param m1: float
        mass_1 of binary
    :param m2: float
        mass_2 of binary
    :param G: float
        gravitational constant
    :return:
        e: float
            new eccentricity
    """
    
    # We take f_ref and convert this into an orbital period 
    # (!! assumes two orbital cycle == 1 GW period)
    P0 = 2.0/f_ref
    
    # Using Kepler's third law we convert the orbital period into a semimajor axis distance
    a0 = period_to_semimajor_a(P0, m1, m2, G)
    
    # Solve for the initial conditions which describe the orbit
    c0 = get_c0(e0, a0, m1, m2, G)
    
    # The period of the orbit at the new reference frequency
    P_new = 2.0/f
    # The new semimajor axis distance of the orbit.
    a_new = period_to_semimajor_a(P_new, m1, m2, G)
    
    # Scalar case
    if scalar:
        e_new = optimize.brentq(semimajor_a_root, 1e-16, 1-1e-16, args=(a_new, c0, m1, m2, G))
    
    # Vector arithmetic
    # Solve for e in a(e) = a_new
    else:
        e_new = optimize.root(semimajor_a_root, e0, args=(a_new, c0, m1, m2, G), tol=1e-10).x
    return e_new







def sigma(e):
    frac = e**(12.0/19.0) / (1-e**2)
    return frac * (1 + (121.0*e**2)/304)**(870.0/2299)

def frequencyEccentricityRoot(eGuess, e0, f, f0):
    sigma0 = sigma(e0)
    sigmaGuess = sigma(eGuess)
    return f - f0 * (sigma0/sigmaGuess)**(3.0/2.0)

def smartShift(f, e0, f_ref, scalar=False):
    if scalar:
        e_new = optimize.brentq(frequencyEccentricityRoot, 1e-16, 1-1e-16, args=(e0, f, f_ref))
    
    # Vector arithmetic
    # Solve for e in a(e) = a_new
    else:
        e_new = optimize.root(frequencyEccentricityRoot, e0, args=(e0, f, f_ref), tol=1e-10).x
    return e_new

Msol = 1.9885 * 10**(30)
print(eccentricity_at_shifted_frequency(f=5, e0=0.2, f_ref=10, m1=2*Msol, m2=1.4*Msol, G=6.67*10**(-11), scalar=True))

