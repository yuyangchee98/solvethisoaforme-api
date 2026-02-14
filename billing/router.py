"""Stripe billing: checkout session creation and webhook handling."""

import os
import logging

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import update

from auth.db import async_session_maker
from auth.models import User
from auth.users import current_active_user

logger = logging.getLogger(__name__)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:4321")

PRICE_IDS = {
    "day_pass": os.environ.get("STRIPE_DAY_PASS_PRICE_ID", ""),
    "individual": os.environ.get("STRIPE_INDIVIDUAL_PRICE_ID", ""),
}

router = APIRouter(prefix="/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    plan: str


@router.post("/create-checkout-session")
async def create_checkout_session(
    body: CheckoutRequest, user: User = Depends(current_active_user)
):
    price_id = PRICE_IDS.get(body.plan)
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid plan")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/agent?checkout=success",
            cancel_url=f"{FRONTEND_URL}/login?checkout=canceled",
            customer_email=user.email,
            metadata={"user_id": str(user.id), "plan": body.plan},
        )
    except stripe.StripeError as e:
        logger.error("Stripe checkout error: %s", e)
        raise HTTPException(status_code=502, detail="Payment service error")

    return {"url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = data.get("metadata", {}).get("user_id")
        plan = data.get("metadata", {}).get("plan")
        if user_id:
            async with async_session_maker() as session:
                await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(
                        stripe_customer_id=data.get("customer"),
                        subscription_id=data.get("subscription"),
                        subscription_status="active",
                        plan_type=plan,
                    )
                )
                await session.commit()

    elif event_type == "customer.subscription.updated":
        sub_id = data.get("id")
        status = data.get("status")
        if sub_id:
            async with async_session_maker() as session:
                await session.execute(
                    update(User)
                    .where(User.subscription_id == sub_id)
                    .values(subscription_status=status)
                )
                await session.commit()

    elif event_type == "customer.subscription.deleted":
        sub_id = data.get("id")
        if sub_id:
            async with async_session_maker() as session:
                await session.execute(
                    update(User)
                    .where(User.subscription_id == sub_id)
                    .values(subscription_status="canceled")
                )
                await session.commit()

    return {"status": "ok"}
