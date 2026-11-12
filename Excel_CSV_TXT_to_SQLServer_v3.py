"""
Streamlit app: Excel / CSV / TXT -> SQL Server Management Studio.

VERSIÓN REDISEÑADA
==================
La aplicación separa completamente dos funcionalidades:

1. CREAR TABLA
   Genera un único archivo .sql con CREATE TABLE a partir de la estructura
   del archivo cargado.

2. INSERTAR DATOS
   Genera todos los archivos .sql necesarios para insertar los datos en una
   tabla ya existente. Los archivos se nombran secuencialmente:
       A1.sql, A2.sql, A3.sql, ...

Reglas:
- Hasta 100,000 filas.
- Hasta 70 columnas.
- Cada archivo A*.sql contiene hasta 10,000 filas.
- Cada sentencia INSERT ... VALUES contiene hasta 1,000 filas.
- No hay slider para configurar el tamaño de INSERT.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import csv
import io
import math
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

MAX_SQL_IDENTIFIER = 128
MAX_DATA_ROWS = 100_000
MAX_DATA_COLUMNS = 70

ROWS_PER_INSERT_FILE = 10_000
ROWS_PER_INSERT_STATEMENT = 1_000

NULL_TEXT = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "<na>",
}

TRUE_TEXT = {
    "true",
    "verdadero",
    "sí",
    "si",
    "yes",
    "1",
}

FALSE_TEXT = {
    "false",
    "falso",
    "no",
    "0",
}

DATE_PATTERN = re.compile(
    r"^(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r")(?:[ T].*)?$"
)


# =============================================================================
# IDENTIFICADORES SQL
# =============================================================================

def normalize_identifier(
    value: str,
    fallback: str = "campo",
) -> str:
    text = unicodedata.normalize(
        "NFKD",
        str(value),
    )
    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )
    text = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        text,
    ).strip("_")

    if not text:
        text = fallback

    if text[0].isdigit():
        text = f"_{text}"

    return text[:MAX_SQL_IDENTIFIER]


def unique_column_names(
    columns,
) -> list[str]:
    used: set[str] = set()
    result: list[str] = []

    for idx, original in enumerate(
        columns,
        start=1,
    ):
        base = normalize_identifier(
            str(original),
            f"columna_{idx}",
        )
        candidate = base
        suffix = 2

        while candidate.casefold() in used:
            extra = f"_{suffix}"
            candidate = (
                f"{base[:MAX_SQL_IDENTIFIER - len(extra)]}"
                f"{extra}"
            )
            suffix += 1

        used.add(candidate.casefold())
        result.append(candidate)

    return result


def quote_identifier(
    name: str,
) -> str:
    return "[" + str(name).replace(
        "]",
        "]]",
    ) + "]"


# =============================================================================
# CARGA DE ARCHIVOS
# =============================================================================

def detect_text_format(
    content: bytes,
) -> tuple[str, str]:
    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    ):
        try:
            sample = content[:20_000].decode(
                encoding
            )
        except UnicodeDecodeError:
            continue

        try:
            delimiter = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t|",
            ).delimiter
        except csv.Error:
            counts = {
                delimiter: sample.count(delimiter)
                for delimiter in ",;\t|"
            }
            delimiter = max(
                counts,
                key=counts.get,
            )

            if counts[delimiter] == 0:
                delimiter = ","

        return encoding, delimiter

    raise ValueError(
        "No se pudo determinar la codificación "
        "del archivo."
    )


def excel_sheets(
    content: bytes,
    filename: str,
) -> list[str]:
    suffix = Path(filename).suffix.lower()

    if suffix not in {
        ".xlsx",
        ".xls",
    }:
        return []

    return pd.ExcelFile(
        BytesIO(content)
    ).sheet_names


def load_file(
    content: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    suffix = Path(filename).suffix.lower()

    meta: dict[str, str] = {
        "tipo": suffix.lstrip("."),
    }

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        df = pd.read_excel(
            BytesIO(content),
            sheet_name=sheet_name or 0,
        )
        meta["hoja"] = (
            sheet_name
            or "Primera hoja"
        )

        return df, meta

    if suffix in {
        ".csv",
        ".txt",
    }:
        encoding, delimiter = detect_text_format(
            content
        )

        df = pd.read_csv(
            BytesIO(content),
            sep=delimiter,
            encoding=encoding,
            low_memory=False,
        )

        meta.update(
            {
                "codificacion": encoding,
                "delimitador": (
                    "\\t"
                    if delimiter == "\t"
                    else delimiter
                ),
            }
        )

        return df, meta

    raise ValueError(
        "Formato no admitido. "
        "Use XLSX, XLS, CSV o TXT."
    )


# =============================================================================
# INFERENCIA DE TIPOS SQL
# =============================================================================

def cleaned_non_null(
    series: pd.Series,
) -> pd.Series:
    s = series.copy()

    if (
        pd.api.types.is_object_dtype(s.dtype)
        or pd.api.types.is_string_dtype(s.dtype)
    ):
        s = s.map(
            lambda value: (
                pd.NA
                if (
                    isinstance(value, str)
                    and value.strip().casefold()
                    in NULL_TEXT
                )
                else value
            )
        )

    return s.dropna()


def infer_object_kind(
    series: pd.Series,
    column_name: str,
) -> tuple[str, str]:
    valid = cleaned_non_null(
        series
    )

    if valid.empty:
        return (
            "NVARCHAR(255)",
            "text",
        )

    text = valid.astype(
        str
    ).str.strip()

    folded = set(
        text.str.casefold().unique()
    )

    if (
        folded
        and folded.issubset(
            TRUE_TEXT | FALSE_TEXT
        )
    ):
        return (
            "BIT",
            "bool_text",
        )

    has_leading_zero = text.str.match(
        r"^[+-]?0\d+$"
    ).any()

    normalized_column_name = (
        unicodedata.normalize(
            "NFKD",
            column_name,
        ).casefold()
    )

    name_hint_id = any(
        token in normalized_column_name
        for token in (
            "id",
            "codigo",
            "code",
            "zip",
            "postal",
            "telefono",
            "phone",
        )
    )

    if (
        not has_leading_zero
        and not name_hint_id
    ):
        numeric = pd.to_numeric(
            text,
            errors="coerce",
        )

        if numeric.notna().all():
            finite = (
                numeric
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if not finite.empty:
                values = finite.to_numpy(
                    float
                )

                if np.isclose(
                    values % 1,
                    0,
                ).all():
                    minimum = finite.min()
                    maximum = finite.max()

                    if (
                        -2_147_483_648
                        <= minimum
                        and maximum
                        <= 2_147_483_647
                    ):
                        return (
                            "INT",
                            "int_text",
                        )

                    return (
                        "BIGINT",
                        "int_text",
                    )

                return (
                    "FLOAT",
                    "float_text",
                )

    sample = text.head(
        200
    )

    date_candidates = sample.map(
        lambda value: bool(
            DATE_PATTERN.match(value)
        )
    )

    date_hint = any(
        token in normalized_column_name
        for token in (
            "fecha",
            "date",
            "time",
            "hora",
        )
    )

    threshold = (
        0.75
        if date_hint
        else 0.90
    )

    if (
        len(sample)
        and date_candidates.mean()
        >= threshold
    ):
        parsed = pd.to_datetime(
            text,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )

        if (
            parsed.notna().mean()
            >= 0.90
        ):
            return (
                "DATETIME2(7)",
                "datetime_text",
            )

    max_length = int(
        text.str.len().max()
    )

    if max_length <= 4_000:
        return (
            f"NVARCHAR({max(1, max_length)})",
            "text",
        )

    return (
        "NVARCHAR(MAX)",
        "text",
    )


def infer_sql_type(
    series: pd.Series,
    column_name: str,
) -> tuple[str, str]:
    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return (
            "BIT",
            "bool",
        )

    if pd.api.types.is_integer_dtype(
        series.dtype
    ):
        valid = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if valid.empty:
            return (
                "BIGINT",
                "int",
            )

        minimum = valid.min()
        maximum = valid.max()

        if (
            -2_147_483_648
            <= minimum
            and maximum
            <= 2_147_483_647
        ):
            return (
                "INT",
                "int",
            )

        return (
            "BIGINT",
            "int",
        )

    if pd.api.types.is_float_dtype(
        series.dtype
    ):
        valid = (
            pd.to_numeric(
                series,
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        if valid.empty:
            return (
                "FLOAT",
                "float",
            )

        values = valid.to_numpy(
            float
        )

        if np.isclose(
            values % 1,
            0,
        ).all():
            minimum = valid.min()
            maximum = valid.max()

            if (
                -2_147_483_648
                <= minimum
                and maximum
                <= 2_147_483_647
            ):
                return (
                    "INT",
                    "int",
                )

            return (
                "BIGINT",
                "int",
            )

        return (
            "FLOAT",
            "float",
        )

    if pd.api.types.is_datetime64_any_dtype(
        series.dtype
    ):
        return (
            "DATETIME2(7)",
            "datetime",
        )

    return infer_object_kind(
        series,
        column_name,
    )


def build_column_specs(
    df: pd.DataFrame,
) -> list[dict]:
    sql_names = unique_column_names(
        df.columns
    )

    specs: list[dict] = []

    for position, (
        original,
        sql_name,
    ) in enumerate(
        zip(
            df.columns,
            sql_names,
        )
    ):
        sql_type, kind = infer_sql_type(
            df.iloc[:, position],
            str(original),
        )

        specs.append(
            {
                "pos": position,
                "original": str(original),
                "sql_name": sql_name,
                "sql_type": sql_type,
                "kind": kind,
            }
        )

    return specs


# =============================================================================
# VALORES SQL
# =============================================================================

def is_null(
    value,
) -> bool:
    if (
        value is None
        or value is pd.NA
    ):
        return True

    try:
        result = pd.isna(
            value
        )

        if isinstance(
            result,
            (
                np.ndarray,
                pd.Series,
            ),
        ):
            return False

        return bool(result)

    except Exception:
        return False


def sql_unicode_string(
    value: str,
) -> str:
    return (
        "N'"
        + str(value)
        .replace(
            "\x00",
            "",
        )
        .replace(
            "'",
            "''",
        )
        + "'"
    )


def sql_literal(
    value,
    kind: str,
) -> str:
    if is_null(
        value
    ):
        return "NULL"

    if kind in {
        "bool",
        "bool_text",
    }:
        if isinstance(
            value,
            str,
        ):
            return (
                "1"
                if value.strip().casefold()
                in TRUE_TEXT
                else "0"
            )

        return (
            "1"
            if bool(value)
            else "0"
        )

    if kind in {
        "int",
        "int_text",
    }:
        try:
            return str(
                int(
                    float(value)
                )
            )
        except Exception:
            return "NULL"

    if kind in {
        "float",
        "float_text",
    }:
        try:
            number = float(
                value
            )

            if math.isfinite(
                number
            ):
                return format(
                    number,
                    ".15g",
                )

            return "NULL"

        except Exception:
            return "NULL"

    if kind in {
        "datetime",
        "datetime_text",
    }:
        try:
            timestamp = pd.Timestamp(
                value
            )

            if pd.isna(
                timestamp
            ):
                return "NULL"

            return sql_unicode_string(
                timestamp.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )
            )

        except Exception:
            return "NULL"

    if isinstance(
        value,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):
        return (
            "0x"
            + bytes(value).hex()
        )

    return sql_unicode_string(
        str(value)
    )


# =============================================================================
# 1. CREAR TABLA
# =============================================================================

def build_create_table_sql(
    specs: list[dict],
    table_name: str,
    schema_name: str = "dbo",
    database_name: str = "",
    drop_if_exists: bool = False,
) -> str:
    """
    Genera únicamente el DDL para crear la tabla.
    NO inserta datos.
    """
    table_name = normalize_identifier(
        table_name,
        "TablaImportada",
    )

    schema_name = normalize_identifier(
        schema_name,
        "dbo",
    )

    database_name = (
        normalize_identifier(
            database_name,
            "",
        )
        if database_name.strip()
        else ""
    )

    q_table = (
        f"{quote_identifier(schema_name)}."
        f"{quote_identifier(table_name)}"
    )

    lines: list[str] = [
        "-- ============================================================",
        "-- SCRIPT 1: CREAR TABLA",
        "-- Generado para SQL Server Management Studio (SSMS)",
        "-- Este archivo NO inserta datos.",
        "-- ============================================================",
        "",
    ]

    if database_name:
        lines.extend(
            [
                f"USE {quote_identifier(database_name)};",
                "GO",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "-- Seleccione la base de datos correcta en SSMS",
                "-- antes de ejecutar este script.",
                "",
            ]
        )

    lines.extend(
        [
            "SET NOCOUNT ON;",
            "SET XACT_ABORT ON;",
            "",
            "-- Mapeo: columna original -> columna SQL (tipo)",
        ]
    )

    for spec in specs:
        lines.append(
            f"-- {spec['original']} -> "
            f"{spec['sql_name']} "
            f"({spec['sql_type']})"
        )

    lines.append("")

    if drop_if_exists:
        safe_object = q_table.replace(
            "'",
            "''",
        )

        lines.extend(
            [
                f"IF OBJECT_ID(N'{safe_object}', N'U') IS NOT NULL",
                f"    DROP TABLE {q_table};",
                "GO",
                "",
            ]
        )

    lines.append(
        f"CREATE TABLE {q_table} ("
    )

    column_lines = [
        (
            f"    {quote_identifier(spec['sql_name'])} "
            f"{spec['sql_type']} NULL"
        )
        for spec in specs
    ]

    lines.append(
        ",\n".join(column_lines)
    )

    lines.extend(
        [
            ");",
            "GO",
            "",
            f"PRINT N'Tabla {schema_name}.{table_name} creada correctamente.';",
            "",
        ]
    )

    return "\n".join(
        lines
    )


# =============================================================================
# 2. INSERTAR DATOS
# =============================================================================

def build_insert_file_sql(
    df_part: pd.DataFrame,
    specs: list[dict],
    table_name: str,
    schema_name: str,
    database_name: str,
    file_number: int,
    total_files: int,
    global_start_row: int,
) -> str:
    """
    Genera un archivo A*.sql.

    Este archivo:
    - NO crea la tabla.
    - Verifica que la tabla exista.
    - Inserta hasta ROWS_PER_INSERT_FILE filas.
    - Divide internamente la carga en sentencias de máximo
      ROWS_PER_INSERT_STATEMENT filas.
    """
    table_name = normalize_identifier(
        table_name,
        "TablaImportada",
    )

    schema_name = normalize_identifier(
        schema_name,
        "dbo",
    )

    database_name = (
        normalize_identifier(
            database_name,
            "",
        )
        if database_name.strip()
        else ""
    )

    q_table = (
        f"{quote_identifier(schema_name)}."
        f"{quote_identifier(table_name)}"
    )

    safe_object = q_table.replace(
        "'",
        "''",
    )

    columns = ", ".join(
        quote_identifier(
            spec["sql_name"]
        )
        for spec in specs
    )

    first_row = (
        global_start_row
        + 1
    )

    last_row = (
        global_start_row
        + len(df_part)
    )

    lines: list[str] = [
        "-- ============================================================",
        f"-- ARCHIVO DE DATOS A{file_number}",
        "-- Solo INSERT. No crea la tabla.",
        f"-- Archivo {file_number} de {total_files}",
        f"-- Filas origen: {first_row:,} a {last_row:,}",
        "-- ============================================================",
        "",
    ]

    if database_name:
        lines.extend(
            [
                f"USE {quote_identifier(database_name)};",
                "GO",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "-- Seleccione la base de datos correcta en SSMS",
                "-- antes de ejecutar este archivo.",
                "",
            ]
        )

    lines.extend(
        [
            "SET NOCOUNT ON;",
            "SET XACT_ABORT ON;",
            "",
            f"IF OBJECT_ID(N'{safe_object}', N'U') IS NULL",
            "BEGIN",
            (
                "    THROW 50001, "
                f"'La tabla {schema_name}.{table_name} no existe. "
                "Ejecute primero el script CREATE TABLE.', 1;"
            ),
            "END;",
            "GO",
            "",
            "BEGIN TRY",
            "    BEGIN TRANSACTION;",
            "",
        ]
    )

    for local_start in range(
        0,
        len(df_part),
        ROWS_PER_INSERT_STATEMENT,
    ):
        local_stop = min(
            local_start
            + ROWS_PER_INSERT_STATEMENT,
            len(df_part),
        )

        global_insert_start = (
            global_start_row
            + local_start
            + 1
        )

        global_insert_stop = (
            global_start_row
            + local_stop
        )

        lines.extend(
            [
                (
                    f"    -- Filas "
                    f"{global_insert_start:,} "
                    f"a {global_insert_stop:,}"
                ),
                (
                    f"    INSERT INTO "
                    f"{q_table} "
                    f"({columns})"
                ),
                "    VALUES",
            ]
        )

        sql_rows: list[str] = []

        chunk = df_part.iloc[
            local_start:local_stop
        ]

        for row in chunk.itertuples(
            index=False,
            name=None,
        ):
            values = [
                sql_literal(
                    row[spec["pos"]],
                    spec["kind"],
                )
                for spec in specs
            ]

            sql_rows.append(
                "        ("
                + ", ".join(values)
                + ")"
            )

        lines.append(
            ",\n".join(
                sql_rows
            )
            + ";"
        )

        lines.append("")

    lines.extend(
        [
            "    COMMIT TRANSACTION;",
            "END TRY",
            "BEGIN CATCH",
            "    IF @@TRANCOUNT > 0",
            "        ROLLBACK TRANSACTION;",
            "    THROW;",
            "END CATCH;",
            "",
            (
                "SELECT COUNT_BIG(*) "
                f"AS FilasActualesEnTabla "
                f"FROM {q_table};"
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def build_all_insert_files(
    df: pd.DataFrame,
    specs: list[dict],
    table_name: str,
    schema_name: str,
    database_name: str,
) -> list[dict]:
    """
    Genera en memoria todos los A*.sql.
    """
    total_rows = len(
        df
    )

    total_files = math.ceil(
        total_rows
        / ROWS_PER_INSERT_FILE
    )

    files: list[dict] = []

    for file_index in range(
        total_files
    ):
        start = (
            file_index
            * ROWS_PER_INSERT_FILE
        )

        stop = min(
            start
            + ROWS_PER_INSERT_FILE,
            total_rows,
        )

        file_number = (
            file_index
            + 1
        )

        filename = (
            f"A{file_number}.sql"
        )

        df_part = df.iloc[
            start:stop
        ]

        sql_text = build_insert_file_sql(
            df_part=df_part,
            specs=specs,
            table_name=table_name,
            schema_name=schema_name,
            database_name=database_name,
            file_number=file_number,
            total_files=total_files,
            global_start_row=start,
        )

        files.append(
            {
                "number": file_number,
                "filename": filename,
                "start_row": start + 1,
                "end_row": stop,
                "row_count": stop - start,
                "content": sql_text.encode(
                    "utf-8"
                ),
            }
        )

    return files


def build_insert_zip(
    insert_files: list[dict],
    table_name: str,
) -> bytes:
    """
    Empaqueta A1.sql, A2.sql, ... en un ZIP.
    """
    memory = io.BytesIO()

    normalized_table = normalize_identifier(
        table_name,
        "TablaImportada",
    )

    with zipfile.ZipFile(
        memory,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        instructions: list[str] = [
            "ARCHIVOS DE INSERT PARA SQL SERVER MANAGEMENT STUDIO",
            "=" * 58,
            "",
            f"Tabla destino: {normalized_table}",
            f"Cantidad de archivos: {len(insert_files)}",
            "",
            "ANTES DE EJECUTAR ESTOS ARCHIVOS:",
            "1. Ejecute primero el script CREATE TABLE generado por la app.",
            "2. Verifique que la tabla fue creada correctamente.",
            "",
            "LUEGO EJECUTE LOS ARCHIVOS EN ESTE ORDEN:",
            "",
        ]

        for item in insert_files:
            archive.writestr(
                item["filename"],
                item["content"],
            )

            instructions.append(
                (
                    f"{item['filename']} | "
                    f"filas "
                    f"{item['start_row']:,} - "
                    f"{item['end_row']:,}"
                )
            )

        archive.writestr(
            "LEEME_ORDEN_INSERTS.txt",
            "\n".join(
                instructions
            )
            + "\n",
        )

    memory.seek(
        0
    )

    return memory.getvalue()


# =============================================================================
# ESTADO STREAMLIT
# =============================================================================

def clear_generated_outputs() -> None:
    for key in (
        "create_sql_bytes",
        "create_sql_filename",
        "insert_files",
        "insert_zip_bytes",
        "insert_zip_filename",
    ):
        st.session_state.pop(
            key,
            None,
        )


def clear_create_output() -> None:
    for key in (
        "create_sql_bytes",
        "create_sql_filename",
    ):
        st.session_state.pop(
            key,
            None,
        )


def clear_insert_outputs() -> None:
    for key in (
        "insert_files",
        "insert_zip_bytes",
        "insert_zip_filename",
    ):
        st.session_state.pop(
            key,
            None,
        )


# =============================================================================
# INTERFAZ
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=(
            "Excel / CSV / TXT "
            "a SQL Server"
        ),
        page_icon="🗄️",
        layout="wide",
    )

    st.title(
        "🗄️ Excel / CSV / TXT "
        "→ SQL Server"
    )

    st.write(
        "La aplicación separa la creación de la tabla "
        "y la carga de los datos en dos procesos independientes."
    )

    st.info(
        "Flujo recomendado: "
        "1) genere y ejecute el script CREATE TABLE; "
        "2) genere los archivos A1.sql, A2.sql, A3.sql... "
        "y ejecútelos en ese orden."
    )

    uploaded = st.file_uploader(
        "Seleccione archivo",
        type=[
            "xlsx",
            "xls",
            "csv",
            "txt",
        ],
    )

    if uploaded is None:
        st.info(
            "Seleccione un archivo XLSX, XLS, CSV o TXT."
        )
        return

    content = uploaded.getvalue()

    try:
        sheets = excel_sheets(
            content,
            uploaded.name,
        )

        sheet_name = (
            st.selectbox(
                "Hoja de Excel",
                sheets,
                index=0,
            )
            if sheets
            else None
        )

        df, meta = load_file(
            content,
            uploaded.name,
            sheet_name,
        )

    except ImportError as error:
        st.error(
            "Falta una dependencia para leer "
            f"el archivo: {error}. "
            "Para archivos .xls instale xlrd."
        )
        return

    except Exception as error:
        st.error(
            "No fue posible importar el archivo: "
            f"{error}"
        )
        return

    if df.shape[1] == 0:
        st.error(
            "El archivo no contiene columnas."
        )
        return

    if len(df) == 0:
        st.error(
            "El archivo no contiene filas de datos."
        )
        return

    if len(df) > MAX_DATA_ROWS:
        st.error(
            f"El archivo contiene {len(df):,} filas. "
            f"El máximo admitido es {MAX_DATA_ROWS:,}."
        )
        return

    if df.shape[1] > MAX_DATA_COLUMNS:
        st.error(
            f"El archivo contiene {df.shape[1]:,} columnas. "
            f"El máximo admitido es {MAX_DATA_COLUMNS:,}."
        )
        return

    specs = build_column_specs(
        df
    )

    total_insert_files = math.ceil(
        len(df)
        / ROWS_PER_INSERT_FILE
    )

    m1, m2, m3, m4 = st.columns(
        4
    )

    m1.metric(
        "Filas",
        f"{len(df):,}",
    )

    m2.metric(
        "Columnas",
        f"{df.shape[1]:,}",
    )

    m3.metric(
        "Archivos INSERT",
        f"{total_insert_files:,}",
    )

    m4.metric(
        "Filas por A*.sql",
        f"{ROWS_PER_INSERT_FILE:,}",
    )

    with st.expander(
        "Detalles de importación"
    ):
        st.write(
            meta
        )

    with st.expander(
        "Vista previa",
        expanded=True,
    ):
        st.dataframe(
            df.head(
                100
            ),
            use_container_width=True,
            hide_index=True,
        )

    mapping = pd.DataFrame(
        [
            {
                "Columna original": spec["original"],
                "Columna SQL": spec["sql_name"],
                "Tipo SQL inferido": spec["sql_type"],
            }
            for spec in specs
        ]
    )

    with st.expander(
        "Tipos SQL inferidos"
    ):
        st.dataframe(
            mapping,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Configuración SQL Server"
    )

    default_table = normalize_identifier(
        Path(
            uploaded.name
        ).stem,
        "TablaImportada",
    )

    c1, c2, c3 = st.columns(
        3
    )

    database_name = c1.text_input(
        "Base de datos (opcional)",
        value="",
        placeholder="Ejemplo: MiBaseDatos",
    )

    schema_name = c2.text_input(
        "Esquema",
        value="dbo",
    )

    table_name = c3.text_input(
        "Nombre de tabla",
        value=default_table,
    )

    if not table_name.strip():
        st.error(
            "Debe indicar un nombre de tabla."
        )
        return

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 1: CREATE TABLE
    # -------------------------------------------------------------------------

    st.header(
        "1. Crear tabla"
    )

    st.write(
        "Genera un único archivo SQL que contiene solamente "
        "la definición de la tabla (`CREATE TABLE`)."
    )

    drop_if_exists = st.checkbox(
        "Eliminar la tabla si ya existe antes de crearla",
        value=False,
        help=(
            "Si se activa, el script CREATE TABLE incluye "
            "DROP TABLE cuando la tabla ya existe."
        ),
    )

    if st.button(
        "Generar script CREATE TABLE",
        type="primary",
        key="generate_create",
    ):
        clear_create_output()

        create_sql = build_create_table_sql(
            specs=specs,
            table_name=table_name,
            schema_name=schema_name,
            database_name=database_name,
            drop_if_exists=drop_if_exists,
        )

        create_filename = (
            f"{normalize_identifier(table_name, 'TablaImportada')}"
            "_CREATE_TABLE.sql"
        )

        st.session_state[
            "create_sql_bytes"
        ] = create_sql.encode(
            "utf-8"
        )

        st.session_state[
            "create_sql_filename"
        ] = create_filename

        st.success(
            "Script CREATE TABLE generado."
        )

    create_bytes = st.session_state.get(
        "create_sql_bytes"
    )

    create_filename = st.session_state.get(
        "create_sql_filename"
    )

    if (
        create_bytes
        and create_filename
    ):
        st.download_button(
            "⬇️ Descargar CREATE TABLE.sql",
            data=create_bytes,
            file_name=create_filename,
            mime="application/sql",
            type="primary",
            key="download_create",
        )

    st.divider()

    # -------------------------------------------------------------------------
    # BLOQUE 2: INSERT DATA
    # -------------------------------------------------------------------------

    st.header(
        "2. Insertar datos"
    )

    st.write(
        "Genera todos los archivos SQL necesarios para cargar "
        "las filas en la tabla ya existente."
    )

    st.info(
        f"Se crearán {total_insert_files:,} archivo(s): "
        "A1.sql, A2.sql, A3.sql... "
        f"Cada archivo contiene hasta {ROWS_PER_INSERT_FILE:,} filas. "
        f"Cada sentencia INSERT contiene automáticamente hasta "
        f"{ROWS_PER_INSERT_STATEMENT:,} filas."
    )

    plan: list[dict] = []

    for file_index in range(
        total_insert_files
    ):
        start = (
            file_index
            * ROWS_PER_INSERT_FILE
        )

        stop = min(
            start
            + ROWS_PER_INSERT_FILE,
            len(df),
        )

        plan.append(
            {
                "Archivo": f"A{file_index + 1}.sql",
                "Filas origen": (
                    f"{start + 1:,} - {stop:,}"
                ),
                "Cantidad": stop - start,
                "Acción": "INSERT",
            }
        )

    with st.expander(
        "Plan de archivos INSERT",
        expanded=True,
    ):
        st.dataframe(
            pd.DataFrame(
                plan
            ),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Generar archivos INSERT",
        type="primary",
        key="generate_inserts",
    ):
        clear_insert_outputs()

        with st.spinner(
            "Generando archivos A1.sql, A2.sql..."
        ):
            insert_files = build_all_insert_files(
                df=df,
                specs=specs,
                table_name=table_name,
                schema_name=schema_name,
                database_name=database_name,
            )

            insert_zip = build_insert_zip(
                insert_files=insert_files,
                table_name=table_name,
            )

        st.session_state[
            "insert_files"
        ] = insert_files

        st.session_state[
            "insert_zip_bytes"
        ] = insert_zip

        st.session_state[
            "insert_zip_filename"
        ] = (
            f"{normalize_identifier(table_name, 'TablaImportada')}"
            "_INSERTS.zip"
        )

        st.success(
            f"Generación completada: "
            f"{len(insert_files)} archivo(s) INSERT."
        )

    insert_files = st.session_state.get(
        "insert_files"
    )

    insert_zip = st.session_state.get(
        "insert_zip_bytes"
    )

    insert_zip_filename = st.session_state.get(
        "insert_zip_filename"
    )

    if insert_files:
        summary = pd.DataFrame(
            [
                {
                    "Archivo": item["filename"],
                    "Filas": (
                        f"{item['start_row']:,} - "
                        f"{item['end_row']:,}"
                    ),
                    "Cantidad": item["row_count"],
                    "Tamaño MB": round(
                        len(item["content"])
                        / (1024 * 1024),
                        2,
                    ),
                }
                for item in insert_files
            ]
        )

        st.subheader(
            "Archivos INSERT generados"
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        if (
            insert_zip
            and insert_zip_filename
        ):
            st.download_button(
                "⬇️ Descargar todos los INSERT (.zip)",
                data=insert_zip,
                file_name=insert_zip_filename,
                mime="application/zip",
                type="primary",
                key="download_insert_zip",
            )

        with st.expander(
            "Descargar archivos INSERT individualmente"
        ):
            for item in insert_files:
                st.download_button(
                    label=(
                        f"⬇️ {item['filename']}"
                    ),
                    data=item["content"],
                    file_name=item["filename"],
                    mime="application/sql",
                    key=(
                        "download_"
                        f"{item['filename']}"
                    ),
                )

        st.success(
            "Orden de ejecución en SSMS: "
            "primero CREATE TABLE; después A1.sql, "
            "A2.sql, A3.sql... en ese orden."
        )


if __name__ == "__main__":
    main()
