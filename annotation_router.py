"""Patent Annotation CRUD API — highlights and notes on patent text."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.users import current_active_user
from sessions.database import get_db

router = APIRouter(prefix="/patents", tags=["annotations"])


# ── Request / response models ────────────────────────────────────────────


class AnnotationCreate(BaseModel):
    id: str | None = None
    section: str
    section_index: int
    paragraph_index: int
    start_offset: int
    end_offset: int
    selected_text: str
    note: str = ""
    color: str = "yellow"
    created_at: str | None = None
    updated_at: str | None = None


class AnnotationUpdate(BaseModel):
    note: str | None = None
    color: str | None = None


class AnnotationOut(BaseModel):
    id: str
    patent_number: str
    section: str
    section_index: int
    paragraph_index: int
    start_offset: int
    end_offset: int
    selected_text: str
    note: str
    color: str
    created_at: str
    updated_at: str


# ── Helpers ───────────────────────────────────────────────────────────────


def _row_to_dict(row: tuple) -> dict:
    keys = [
        "id", "user_id", "patent_number", "section", "section_index",
        "paragraph_index", "start_offset", "end_offset", "selected_text",
        "note", "color", "created_at", "updated_at",
    ]
    return dict(zip(keys, row))


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.get("/{publication_number}/annotations", response_model=list[AnnotationOut])
async def list_annotations(publication_number: str, user=Depends(current_active_user)):
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM patent_annotations WHERE user_id = ? AND patent_number = ? ORDER BY created_at",
        (str(user.id), publication_number),
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


@router.post("/{publication_number}/annotations", response_model=AnnotationOut, status_code=201)
async def create_annotation(
    publication_number: str,
    body: AnnotationCreate,
    user=Depends(current_active_user),
):
    db = await get_db()
    ann_id = body.id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await db.execute(
        """INSERT INTO patent_annotations
           (id, user_id, patent_number, section, section_index, paragraph_index,
            start_offset, end_offset, selected_text, note, color, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ann_id, str(user.id), publication_number,
            body.section, body.section_index, body.paragraph_index,
            body.start_offset, body.end_offset, body.selected_text,
            body.note, body.color,
            body.created_at or now, body.updated_at or now,
        ),
    )
    await db.commit()
    return {
        "id": ann_id, "patent_number": publication_number,
        "section": body.section, "section_index": body.section_index,
        "paragraph_index": body.paragraph_index,
        "start_offset": body.start_offset, "end_offset": body.end_offset,
        "selected_text": body.selected_text, "note": body.note, "color": body.color,
        "created_at": body.created_at or now, "updated_at": body.updated_at or now,
    }


@router.patch("/{publication_number}/annotations/{annotation_id}", response_model=AnnotationOut)
async def update_annotation(
    publication_number: str,
    annotation_id: str,
    body: AnnotationUpdate,
    user=Depends(current_active_user),
):
    db = await get_db()
    cursor = await db.execute(
        "SELECT * FROM patent_annotations WHERE id = ? AND user_id = ?",
        (annotation_id, str(user.id)),
    )
    row = await cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Annotation not found")

    existing = _row_to_dict(row)
    now = datetime.now(timezone.utc).isoformat()
    new_note = body.note if body.note is not None else existing["note"]
    new_color = body.color if body.color is not None else existing["color"]

    await db.execute(
        "UPDATE patent_annotations SET note = ?, color = ?, updated_at = ? WHERE id = ? AND user_id = ?",
        (new_note, new_color, now, annotation_id, str(user.id)),
    )
    await db.commit()
    existing.update(note=new_note, color=new_color, updated_at=now)
    return existing


@router.delete("/{publication_number}/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    publication_number: str,
    annotation_id: str,
    user=Depends(current_active_user),
):
    db = await get_db()
    cursor = await db.execute(
        "DELETE FROM patent_annotations WHERE id = ? AND user_id = ?",
        (annotation_id, str(user.id)),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="Annotation not found")


class BulkAnnotation(BaseModel):
    id: str | None = None
    patent_number: str
    section: str
    section_index: int
    paragraph_index: int
    start_offset: int
    end_offset: int
    selected_text: str
    note: str = ""
    color: str = "yellow"
    created_at: str | None = None
    updated_at: str | None = None


class BulkImportRequest(BaseModel):
    annotations: list[BulkAnnotation]


class BulkImportResponse(BaseModel):
    imported: int


@router.post("/annotations/import", response_model=BulkImportResponse, status_code=200)
async def bulk_import_annotations(body: BulkImportRequest, user=Depends(current_active_user)):
    db = await get_db()
    uid = str(user.id)
    now = datetime.now(timezone.utc).isoformat()
    imported = 0
    for a in body.annotations:
        ann_id = a.id or str(uuid.uuid4())
        cursor = await db.execute(
            """INSERT OR IGNORE INTO patent_annotations
               (id, user_id, patent_number, section, section_index, paragraph_index,
                start_offset, end_offset, selected_text, note, color, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ann_id, uid, a.patent_number,
                a.section, a.section_index, a.paragraph_index,
                a.start_offset, a.end_offset, a.selected_text,
                a.note, a.color,
                a.created_at or now, a.updated_at or now,
            ),
        )
        imported += cursor.rowcount
    await db.commit()
    return {"imported": imported}
