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

PROJECT_ID = "solvethisoaforme"

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

    # Reuse existing Stripe customer if available
    customer_kwargs = {}
    if user.stripe_customer_id:
        customer_kwargs["customer"] = user.stripe_customer_id
    else:
        customer_kwargs["customer_email"] = user.email

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            allow_promotion_codes=True,
            success_url=f"{FRONTEND_URL}/agent?checkout=success",
            cancel_url=f"{FRONTEND_URL}/login?checkout=canceled",
            **customer_kwargs,
            metadata={
                "user_id": str(user.id),
                "plan": body.plan,
                "project": PROJECT_ID,
            },
            subscription_data={
                "metadata": {
                    "user_id": str(user.id),
                    "project": PROJECT_ID,
                },
            },
        )
    except stripe.StripeError as e:
        logger.error("Stripe checkout error: %s", e)
        raise HTTPException(status_code=502, detail="Payment service error")

    return {"url": session.url}


@router.post("/create-portal-session")
async def create_portal_session(user: User = Depends(current_active_user)):
    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")

    try:
        session = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=f"{FRONTEND_URL}/agent",
        )
    except stripe.StripeError as e:
        logger.error("Stripe portal error: %s", e)
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

    # Skip events not meant for this project
    metadata = data.get("metadata", {})
    if metadata.get("project") != PROJECT_ID:
        return {"status": "skipped"}

    if event_type == "checkout.session.completed":
        user_id = metadata.get("user_id")
        plan = metadata.get("plan")
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
