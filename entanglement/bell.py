from .parse import CHSH, CHSHCounts
from .statistics import nsigma
from .util import perp, Sign, phi_name


def expectation(counts: CHSHCounts, alpha, beta) -> float:
    alpha_p = perp(alpha)
    beta_p = perp(beta)
    c_a_b = counts[alpha, beta].c
    c_ap_bp = counts[alpha_p, beta_p].c
    c_ap_b = counts[alpha_p, beta].c
    c_a_bp = counts[alpha, beta_p].c
    exp = (c_a_b + c_ap_bp - c_ap_b - c_a_bp)/(c_a_b + c_ap_bp + c_ap_b + c_a_bp)
    # print(f"expectation for alpha: {alpha}, beta: {beta}, exp: {exp:.3}")
    return exp


def expectation_err(counts: CHSHCounts, alpha, beta) -> float:
    alpha_p = perp(alpha)
    beta_p = perp(beta)
    c_a_b = counts[alpha, beta].c
    c_ap_bp = counts[alpha_p, beta_p].c
    c_ap_b = counts[alpha_p, beta].c
    c_a_bp = counts[alpha, beta_p].c
    exp_err = (
        2
        * ((c_a_b + c_ap_bp) * (c_a_bp + c_ap_b))
        / (c_a_b + c_ap_bp + c_a_bp + c_ap_b)**2
        * (1/(c_a_b + c_ap_bp) + 1/(c_a_bp + c_ap_b))**(1/2)
    )
    return exp_err


def s_corr(sign: Sign, counts: CHSHCounts, alpha_1, alpha_2, beta_1, beta_2) -> float:
    s = (
        + expectation(counts, alpha_1, beta_1)
        + expectation(counts, alpha_2, beta_1) * float(sign)
        - expectation(counts, alpha_1, beta_2)
        + expectation(counts, alpha_2, beta_2) * float(sign)
    )
    return s


def s_corr_err(counts: CHSHCounts, alpha_1, alpha_2, beta_1, beta_2) -> float:
    s_err = sum(
        expectation_err(counts, alpha, beta)**2
        for alpha in [alpha_1, alpha_2]
        for beta in [beta_1, beta_2]
    )**(1/2)
    return s_err


def run_bell(sign: Sign, path: str):
    print(f"analyzing {path} as {phi_name(sign)}")
    counts = CHSH.from_file(path).counts

    # test angles as fractions of 2pi:
    circle = 360
    alpha_1 = 0 * circle
    alpha_2 = 1/8 * circle
    beta_1 = 1/16 * circle
    beta_2 = (1/16 + 1/8) * circle

    s = s_corr(sign, counts, alpha_1, alpha_2, beta_1, beta_2)
    s_err = s_corr_err(counts, alpha_1, alpha_2, beta_1, beta_2)
    ns = nsigma(s, s_err, 2)
    print(f"s: {s:.5} ± {s_err:.2} ({s_err/s:.2%}) nsigma: {int(ns)}")
