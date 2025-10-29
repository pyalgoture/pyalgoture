from datetime import date


class FundTracker:
    def __init__(self, nav_per_share: float, outstanding_shares: float | None = None, aum: float | None = None):
        self.nav_per_share: float = nav_per_share
        self.aum: float = 0.0
        self.outstanding_shares: float = 0.0
        self.pnl: float = 0.0
        self.snr: list = []

        if outstanding_shares and aum:
            raise ValueError("Either outstanding_shares or aum must be provided.")
        if outstanding_shares:
            self.outstanding_shares = outstanding_shares
            self.aum = self.nav_per_share * self.outstanding_shares
        if aum:
            self.aum = aum
            self.outstanding_shares = self.aum / self.nav_per_share if self.nav_per_share else 0.0

    def __repr__(self) -> str:
        # return f"AUM: ${round(self.aum, 2)}\n" f"NAV(per share): ${round(self.nav_per_share, 2)}\n" f"Outstanding shares: {round(self.outstanding_shares, 2)}\n"
        return f"AUM: ${round(self.aum, 2)}; NAV(per share): ${round(self.nav_per_share, 2)}; Outstanding shares: {round(self.outstanding_shares, 2)}\n"

    def subscribe(self, amount: float, dt: date | None = None) -> dict:
        self.aum += amount
        self.outstanding_shares = self.aum / self.nav_per_share if self.nav_per_share else 0.0
        data = {
            "date": dt if dt else date.today(),
            "amount": amount,
            "price": self.nav_per_share,
            "unit": amount / self.nav_per_share if self.nav_per_share else 0.0,
            "type": "subscription",
        }
        self.snr.append(data)
        return data

    def redeem(self, amount: float, dt: date | None = None) -> dict:
        if amount > self.aum:
            raise ValueError("Redemption amount cannot exceed AUM.")
        self.aum -= amount
        self.outstanding_shares = self.aum / self.nav_per_share if self.nav_per_share else 0.0
        data = {
            "date": dt if dt else date.today(),
            "amount": amount,
            "price": self.nav_per_share,
            "unit": amount / self.nav_per_share if self.nav_per_share else 0.0,
            "type": "redemption",
        }
        self.snr.append(data)
        return data

    def update_nav(self, new_nav_per_share: float) -> None:
        self.nav_per_share = new_nav_per_share
        self.aum = self.nav_per_share * self.outstanding_shares

    def update_pnl(self, pnl: float) -> None:
        self.pnl = pnl
        self.aum += pnl
        self.nav_per_share = self.aum / self.outstanding_shares if self.outstanding_shares else 0.0

    def get_aum(self) -> float:
        return self.aum

    def get_nav_per_share(self) -> float:
        return self.nav_per_share

    def get_outstanding_shares(self) -> float:
        return self.outstanding_shares

    def get_pnl(self) -> float:
        return self.pnl

    def get_dict(self) -> dict:
        return {
            "nav_per_share": self.nav_per_share,
            "aum": self.aum,
            "outstanding_shares": self.outstanding_shares,
            "pnl": self.pnl,
            "snr": self.snr,
        }


if __name__ == "__main__":
    # fund = FundTracker(nav_per_share=100, outstanding_shares=1000)
    # fund = FundTracker(nav_per_share=100, aum=100_000)
    fund = FundTracker(nav_per_share=100, aum=0)
    print("========= initialization =========")
    print("AUM:", fund.get_aum())
    print("NAV(per share):", fund.get_nav_per_share())
    print("Outstanding shares:", fund.get_outstanding_shares())

    print("========= subscription =========")
    fund.subscribe(50_000)  # Subscribe $500,000 into the fund
    print(fund)

    print("========= update pnl =========")
    fund.update_pnl(11111)
    print(fund)

    print("========= redemption =========")
    fund.redeem(10000)  # Redeem $10,000 from the fund
    print(fund)

    print("----" * 8)
    print(fund)
    print(fund.snr)
